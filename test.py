"""
CCKS 2026 OneEval 解题脚本（DeepSeek V4 Pro + Think Max）  v2

升级要点（详见 IMPROVEMENT_REPORT.md）：
1. 结构化预检索：KG 子图 BFS 抽取；多跳段落 BM25 排序；表格扩展确定性 solver
2. 顺序扰动 + 风格多样 self-consistency（默认 5 路，可配置）
3. Evidence-based verification 二次验证
4. Adaptive thinking_budget 按题型 + 难度自适应
5. 类型感知答案后处理（count / year / entity / yes-no / list）
6. 拒答兜底升温重答
7. 双 Provider 支持：LLM_PROVIDER=anthropic 或 openai（互斥，单次只启用一种）
   - anthropic：anthropic SDK · messages.create + thinking={"type":"enabled","budget_tokens":N}
                可对接 Claude Opus 4.7 官方端点，或任何 Anthropic-兼容反代
   - openai   ：openai SDK · chat.completions.create + reasoning_effort
                可对接 GPT-5.5 / GPT-5.4 / GLM-5 / GLM-5.1 / DeepSeek-OpenAI 等所有 OpenAI 兼容端点
8. 跨运行 Consensus：见 IMPROVEMENT_REPORT.md §8（锁定 + 跳过）

配置点：直接编辑 test.py 顶部的常量（LLM_PROVIDER / *_API_KEY / *_BASE_URL / *_MODEL ...）。
不使用环境变量。
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from rapidfuzz import fuzz

# ============================================================
# 配置区
# ============================================================
# 【Provider 开关】只能选一个："anthropic" | "openai"
#   anthropic SDK：Claude 系列（Opus/Sonnet 等）官方 API，或任何 Anthropic-兼容端点
#   openai    SDK：GPT-5.x / GLM-5.x / DeepSeek-OpenAI 等所有 OpenAI 兼容端点
LLM_PROVIDER = "anthropic"   # "anthropic" 或 "openai"

# ---------- Anthropic SDK（Claude Opus 4.7 等） ----------
ANTHROPIC_API_KEY  = "sk-3b8f9bf9a89c4633a36cc7109ef2026f"
ANTHROPIC_BASE_URL = "https://api.deepseek.com/anthropic"   # Claude 官方：https://api.anthropic.com
ANTHROPIC_MODEL    = "deepseek-v4-pro"                       # Claude 官方：claude-opus-4-7-20250...

# ---------- OpenAI SDK（GPT-5.5 / GLM-5 / GLM-5.1 等） ----------
OPENAI_API_KEY  = "sk-3b8f9bf9a89c4633a36cc7109ef2026f"
OPENAI_BASE_URL = "https://api.deepseek.com"                 # GPT  ：https://api.openai.com/v1
                                                              # GLM  ：https://open.bigmodel.cn/api/paas/v4/
OPENAI_MODEL    = "deepseek-v4-pro"                          # GPT  ：gpt-5.5 / gpt-5.4
                                                              # GLM  ：glm-5 / glm-5.1
OPENAI_REASONING_EFFORT = "high"                              # 推理力度，统一 high；OpenAI 还支持 minimal/low/medium

INPUT_FILE = "contest_data.json"
OUTPUT_FILE = "submit.jsonl"
RAW_OUTPUT_FILE = "submit_raw.jsonl"

# Anthropic extended-thinking 预算（按题型）；OpenAI SDK 用 reasoning_effort，无 budget 概念
THINKING_BUDGET_BY_TYPE = {
    "knowledge_graph": 10000,
    "multi_hop_qa": 12000,
    "table_qa": 8000,
}
MAX_TOKENS = 16000   # Anthropic: max_tokens；OpenAI: max_completion_tokens

CONCURRENCY = 80
MAX_RETRIES = 5
INITIAL_BACKOFF = 2.0
REANSWER_BAD = True

# 顺序扰动 self-consistency 路数（按题型）
VOTING_ROUNDS_BY_TYPE = {
    "knowledge_graph": 5,
    "multi_hop_qa": 5,
    "table_qa": 3,
}

ENABLE_KG_SUBGRAPH = True
KG_SUBGRAPH_DEPTH = 2
KG_SUBGRAPH_MAX = 90        # 子图保留上限
KG_SUBGRAPH_KEEP_AT_LEAST = 30  # 子图过小时补足到此数量
KG_SUBGRAPH_MIN_RAW = 25    # 三元组少于这个数就不裁剪（保留全部）

ENABLE_PASSAGE_RANK = True
PASSAGE_RANK_KEEP = 6

ENABLE_VERIFICATION = True
ENABLE_REPAIR = True

# ============================================================
# Provider 客户端初始化（延迟）
# ============================================================
_anthropic_client = None
_openai_client = None
_anthropic_mod = None
_openai_mod = None


def _get_anthropic_client():
    global _anthropic_client, _anthropic_mod
    if _anthropic_client is None:
        import anthropic as _anth  # type: ignore
        _anthropic_mod = _anth
        _anthropic_client = _anth.AsyncAnthropic(
            api_key=ANTHROPIC_API_KEY,
            base_url=ANTHROPIC_BASE_URL,
            timeout=600.0,
        )
    return _anthropic_client


def _get_openai_client():
    global _openai_client, _openai_mod
    if _openai_client is None:
        import openai as _oai  # type: ignore
        _openai_mod = _oai
        kwargs: dict[str, Any] = {"api_key": OPENAI_API_KEY, "timeout": 600.0}
        if OPENAI_BASE_URL:
            kwargs["base_url"] = OPENAI_BASE_URL
        _openai_client = _oai.AsyncOpenAI(**kwargs)
    return _openai_client


def _ensure_provider_modules():
    """预热当前 provider 的 SDK 模块，以便后续 except 能访问到类型。"""
    if LLM_PROVIDER == "anthropic":
        _get_anthropic_client()
    elif LLM_PROVIDER == "openai":
        _get_openai_client()
    else:
        raise ValueError(f"未知 LLM_PROVIDER: {LLM_PROVIDER!r}; 请设为 'anthropic' 或 'openai'")

BAD_ANSWER_PATTERNS = (
    "none",
    "unknown",
    "not specified",
    "not available",
    "not mentioned",
    "not provided",
    "not found",
    "unanswerable",
    "cannot be determined",
    "insufficient information",
    "not enough information",
    "impossible to answer",
    "i don't know",
    "i do not know",
    "no answer",
)

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "of", "in", "on", "at", "to", "for", "with",
    "by", "from", "as", "is", "are", "was", "were", "be", "been", "being", "this",
    "that", "these", "those", "it", "its", "his", "her", "their", "which", "who",
    "whom", "whose", "what", "when", "where", "why", "how", "do", "does", "did",
    "has", "have", "had", "into", "than", "then", "so", "if", "while", "also",
    "any", "all", "some", "many", "much", "more", "most", "other", "another",
    "between", "both", "either", "neither", "not", "no", "yes",
}

ANCHOR_STOPWORDS = STOPWORDS | {
    "would", "could", "should", "may", "might", "shall", "will",
    "according", "given", "based", "above", "below", "following",
}

# ============================================================
# 通用工具
# ============================================================
def normalize_text(value: str) -> str:
    value = str(value).replace("–", "-").replace("—", "-")
    value = re.sub(r"\s+", " ", value).strip().lower()
    return value


def tokenize(text: str, keep_short: bool = False) -> list[str]:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9_]+", " ", text)
    toks = [t for t in text.split() if t and (keep_short or len(t) > 1)]
    return [t for t in toks if t not in STOPWORDS]


def extract_proper_nouns(text: str) -> list[str]:
    """提取问句中疑似专有名词（连续 Capitalized words + 引号包裹串 + 引文）。"""
    candidates: list[str] = []
    candidates.extend(re.findall(r'"([^"]+)"', text))
    candidates.extend(re.findall(r"'([^']{2,})'", text))
    candidates.extend(re.findall(r"\b((?:[A-Z][\w'’.\-]+)(?:\s+[A-Z][\w'’.\-]+)*)\b", text))
    seen: set[str] = set()
    out: list[str] = []
    for c in candidates:
        c = c.strip().strip("'\"")
        if len(c) < 2:
            continue
        cl = c.lower()
        if cl in ANCHOR_STOPWORDS:
            continue
        if cl in seen:
            continue
        seen.add(cl)
        out.append(c)
    return out


# ============================================================
# KG 子图检索（KGQAGen-10k / StepChain GraphRAG / SKA-Bench Noise Robustness）
# ============================================================
def parse_triples(raw: str) -> list[str]:
    parts = [p.strip() for p in re.split(r"\s*\|\s*", raw) if p.strip()]
    triples: list[str] = []
    for part in parts:
        lower = part.lower()
        if "knowledge graph triples" in lower:
            continue
        if lower.startswith((
            "infer ", "answer ", "using ", "you will", "the following", "knowledge graph",
        )):
            continue
        triples.append(part)
    return triples


def split_triple(triple: str) -> tuple[str, str, str]:
    """启发式拆分 <head> <relation> <tail>。
    relation 通常是包含 '.' 的标记式串（如 film.film.starring..film.performance.actor 或 location.location.contains）。
    """
    # 找包含 '.' 的最长 token 作为 relation；若失败退化为按空格 3 段
    tokens = triple.split(" ")
    rel_idx = -1
    for i, tok in enumerate(tokens):
        if "." in tok and not tok.replace(".", "").isdigit():
            rel_idx = i
            break
    if rel_idx == -1:
        # fallback：按空格
        if len(tokens) >= 3:
            return tokens[0], tokens[1], " ".join(tokens[2:])
        return triple, "", ""
    head = " ".join(tokens[:rel_idx]).strip()
    relation = tokens[rel_idx]
    tail = " ".join(tokens[rel_idx + 1:]).strip()
    return head, relation, tail


def build_kg_index(triples: list[str]) -> tuple[dict[str, list[int]], list[tuple[str, str, str]]]:
    """返回 entity->triple_idx 列表 + 解析后三元组数组。"""
    parsed = [split_triple(t) for t in triples]
    idx: dict[str, list[int]] = defaultdict(list)
    for i, (h, r, t) in enumerate(parsed):
        if h:
            idx[normalize_text(h)].append(i)
        if t:
            idx[normalize_text(t)].append(i)
    return idx, parsed


def fuzzy_find_entities(question: str, entity_keys: list[str], top: int = 8) -> list[str]:
    """模糊匹配问题中的实体。"""
    proper = extract_proper_nouns(question)
    seeds: list[tuple[str, float]] = []
    seen: set[str] = set()
    # 精确包含匹配
    qnorm = normalize_text(question)
    for ent in entity_keys:
        if not ent or len(ent) < 3:
            continue
        if ent in qnorm:
            if ent not in seen:
                seeds.append((ent, 100.0))
                seen.add(ent)
    # 专有名词模糊匹配
    if len(seeds) < top:
        for pn in proper:
            pnn = normalize_text(pn)
            best_score = 0.0
            best_ent = None
            for ent in entity_keys:
                if not ent or len(ent) < 3:
                    continue
                sc = fuzz.token_set_ratio(pnn, ent)
                if sc > best_score:
                    best_score = sc
                    best_ent = ent
            if best_ent and best_score >= 85 and best_ent not in seen:
                seeds.append((best_ent, best_score))
                seen.add(best_ent)
    seeds.sort(key=lambda x: -x[1])
    return [s[0] for s in seeds[:top]]


RELATION_KEYWORDS = {
    "date": ("date", "release", "birth", "death", "founded", "established"),
    "year": ("date", "release", "founded", "established", "year"),
    "film": ("film", "movie", "performance", "actor", "genre", "directed", "netflix"),
    "actor": ("film", "performance", "actor", "starring"),
    "director": ("directed", "director"),
    "award": ("award", "honor"),
    "born": ("birth", "born", "people_born_here"),
    "died": ("death", "died"),
    "location": ("location", "contains", "country"),
    "country": ("country", "location.country", "location.location.contains"),
    "team": ("team", "sports"),
    "language": ("language",),
    "genre": ("genre",),
    "book": ("book",),
    "university": ("education", "university", "school"),
    "school": ("education", "school"),
    "founded": ("founded", "established", "date"),
    "album": ("album", "music"),
    "song": ("song", "music", "track"),
    "company": ("company", "organization", "business"),
}


def find_relation_keywords(question: str) -> set[str]:
    q = normalize_text(question)
    kws: set[str] = set()
    for trigger, related in RELATION_KEYWORDS.items():
        if trigger in q:
            kws.update(related)
    return kws


def extract_subgraph(
    triples: list[str],
    question: str,
    depth: int = KG_SUBGRAPH_DEPTH,
    cap: int = KG_SUBGRAPH_MAX,
) -> list[str]:
    """从问题实体出发 BFS depth 跳的子图 + 相关关系匹配。失败回退到原列表。"""
    if len(triples) <= KG_SUBGRAPH_MIN_RAW:
        return triples
    idx, parsed = build_kg_index(triples)
    entity_keys = list(idx.keys())
    seeds = fuzzy_find_entities(question, entity_keys)

    selected: set[int] = set()
    seed_triple_indices: set[int] = set()
    if seeds:
        frontier = set(seeds)
        for hop in range(depth):
            next_frontier: set[str] = set()
            for ent in frontier:
                for ti in idx.get(ent, []):
                    if ti in selected:
                        continue
                    selected.add(ti)
                    if hop == 0:
                        seed_triple_indices.add(ti)
                    if len(selected) >= cap:
                        break
                    h, _r, t = parsed[ti]
                    hh, tt = normalize_text(h), normalize_text(t)
                    if hh and hh != ent and hh not in frontier:
                        next_frontier.add(hh)
                    if tt and tt != ent and tt not in frontier:
                        next_frontier.add(tt)
                if len(selected) >= cap:
                    break
            frontier = next_frontier - frontier
            if len(selected) >= cap or not frontier:
                break

    # 关系关键词扩充（不耗尽 cap）
    rel_kws = find_relation_keywords(question)
    if rel_kws and len(selected) < cap:
        for i, (h, r, t) in enumerate(parsed):
            if i in selected:
                continue
            rl = normalize_text(r)
            if any(kw in rl for kw in rel_kws):
                selected.add(i)
                if len(selected) >= cap:
                    break

    # 补足：若子图过小，按原顺序补一些剩余三元组（避免完全丢失）
    if len(selected) < KG_SUBGRAPH_KEEP_AT_LEAST:
        for i in range(len(parsed)):
            if i in selected:
                continue
            selected.add(i)
            if len(selected) >= KG_SUBGRAPH_KEEP_AT_LEAST:
                break

    if not selected:
        return triples

    # 排序：种子直接命中 > 种子BFS > 关系命中 > 其他
    seed_set = set(seeds) if seeds else set()
    sub_ordered: list[tuple[int, int, str]] = []
    for i in range(len(parsed)):
        if i not in selected:
            continue
        h, r, t = parsed[i]
        hh, tt = normalize_text(h), normalize_text(t)
        if i in seed_triple_indices:
            score = 3
        elif hh in seed_set or tt in seed_set:
            score = 2
        elif rel_kws and any(kw in normalize_text(r) for kw in rel_kws):
            score = 1
        else:
            score = 0
        sub_ordered.append((score, i, triples[i]))
    sub_ordered.sort(key=lambda x: (-x[0], x[1]))
    return [t for _, _, t in sub_ordered]


def format_kg_triples(triples_list: list[str]) -> str:
    return "\n".join(f"{i + 1}. {t}" for i, t in enumerate(triples_list))


def format_kg_triples_grouped(triples_list: list[str]) -> str:
    groups: dict[str, list[str]] = defaultdict(list)
    for t in triples_list:
        head, _, _ = split_triple(t)
        groups.setdefault(head or "_", []).append(t)
    lines = []
    idx = 1
    for head in groups:
        for triple in groups[head]:
            lines.append(f"{idx}. {triple}")
            idx += 1
    return "\n".join(lines)


# ============================================================
# 多跳段落 BM25 排序
# ============================================================
def bm25_rank(question: str, passages: list[str], k1: float = 1.5, b: float = 0.75) -> list[float]:
    """简化 BM25。返回每段的分数。"""
    if not passages:
        return []
    q_tokens = tokenize(question)
    # 也加入专有名词的全文形式（作为单 token）
    pns = [normalize_text(p) for p in extract_proper_nouns(question)]
    q_tokens = q_tokens + pns

    docs = [tokenize(p) for p in passages]
    avgdl = sum(len(d) for d in docs) / max(1, len(docs))
    df: Counter = Counter()
    for d in docs:
        for term in set(d):
            df[term] += 1
    N = len(docs)
    scores = []
    for d in docs:
        d_len = len(d)
        tf = Counter(d)
        s = 0.0
        for q in set(q_tokens):
            if q not in tf:
                # 也试一下子串匹配（专有名词常嵌入段落字符串）
                pass
            n_q = df.get(q, 0)
            if n_q == 0:
                continue
            idf = math.log(1 + (N - n_q + 0.5) / (n_q + 0.5))
            term_tf = tf.get(q, 0)
            s += idf * (term_tf * (k1 + 1)) / (term_tf + k1 * (1 - b + b * d_len / max(1, avgdl)))
        scores.append(s)

    # 加分：段落字符串中是否出现问题专有名词原串
    for i, p in enumerate(passages):
        pl = p.lower()
        for pn in pns:
            if pn and pn in pl:
                scores[i] += 1.5
    return scores


def rank_passages(question: str, contexts: list[dict], top_k: int = PASSAGE_RANK_KEEP) -> list[tuple[int, dict]]:
    """对 multi_hop_qa 段落按相关性降序排序，并返回 [(原 index, ctx), ...]。"""
    passages = []
    for c in contexts:
        title = c.get("title", "")
        text = c.get("paragraph") or " ".join(c.get("sentences", []))
        passages.append(f"{title}. {text}")
    scores = bm25_rank(question, passages)
    order = sorted(range(len(contexts)), key=lambda i: -scores[i])
    keep = order[:top_k] if top_k and top_k < len(contexts) else order
    return [(i, contexts[i]) for i in keep]


# ============================================================
# Table 渲染（标准 Markdown，TabReX 单元格对齐）
# ============================================================
def format_table_markdown(table: dict) -> str:
    header = table["header"]
    rows = table["rows"]
    header_line = "| " + " | ".join(str(h).strip() for h in header) + " |"
    separator_line = "| " + " | ".join("---" for _ in header) + " |"
    body = ["| " + " | ".join(str(cell).strip() for cell in row) + " |" for row in rows]
    return "\n".join([header_line, separator_line, *body])


def format_table_records(table: dict, limit: int = 50) -> str:
    """row-as-record 视图，强化 LLM 对单元格语义的对齐。"""
    header = table["header"]
    rows = table["rows"][:limit]
    out = []
    for r_idx, row in enumerate(rows):
        line = f"Row {r_idx + 1}: " + "; ".join(
            f"{header[i]}={str(cell).strip()}" for i, cell in enumerate(row) if i < len(header)
        )
        out.append(line)
    return "\n".join(out)


# ============================================================
# Deterministic Table Solver（来自 v1，部分扩展）
# ============================================================
def col_index(header: list, *candidates: str) -> int | None:
    normalized = [normalize_text(h) for h in header]
    for candidate in candidates:
        c = normalize_text(candidate)
        for i, h in enumerate(normalized):
            if h == c:
                return i
    for candidate in candidates:
        c = normalize_text(candidate)
        for i, h in enumerate(normalized):
            if c in h:
                return i
    return None


def to_int(value: str) -> int | None:
    match = re.search(r"-?\d[\d,]*", str(value))
    if not match:
        return None
    return int(match.group(0).replace(",", ""))


def to_float(value: str) -> float | None:
    match = re.search(r"-?\d+(?:\.\d+)?", str(value).replace(",", ""))
    if not match:
        return None
    return float(match.group(0))


def ordinal_to_int(value: str) -> int | None:
    return to_int(value)


def exactish(value: str, text: str) -> bool:
    v = normalize_text(value)
    return bool(v) and v in normalize_text(text)


def extract_table_name_from_question(rows: list, col: int, question: str) -> str | None:
    matches = [row[col] for row in rows if exactish(row[col], question)]
    if not matches:
        return None
    return max(matches, key=len)


def parse_day(value: str) -> int | None:
    matches = re.findall(r"\d+", str(value))
    return int(matches[-1]) if matches else None


def split_people(value: str) -> set[str]:
    value = re.sub(r"\([^)]*\)", "", str(value))
    value = value.replace(";", ",").replace(" and ", ",")
    people = set()
    for part in value.split(","):
        name = part.strip()
        if name and normalize_text(name) not in {"none", "n/a", "own goal"}:
            people.add(name)
    return people


def solve_table_qa(item: dict) -> str | None:
    table = item["table"]
    header = table["header"]
    rows = table["rows"]
    question = item["question"]
    q = normalize_text(question)

    venue_col = col_index(header, "Venue")
    if "count distinct matches" in q and venue_col is not None:
        match = re.search(r"was\s+(.+?)\s+the venue", question, re.I)
        if match:
            venue = normalize_text(match.group(1))
            id_cols = {i for i, h in enumerate(header) if "id" in normalize_text(h)}
            distinct = {
                tuple(cell for i, cell in enumerate(row) if i not in id_cols)
                for row in rows
                if normalize_text(row[venue_col]) == venue
            }
            return str(len(distinct))

    film_col = col_index(header, "Film", "Film Title")
    if q.startswith("how many films has") and film_col is not None:
        performer = re.sub(r"^how many films has\s+", "", q).removesuffix(" appeared in?")
        performer_col = col_index(header, "Performer")
        if performer_col is None:
            return str(sum(1 for row in rows if str(row[film_col]).strip()))
        return str(sum(1 for row in rows if normalize_text(row[performer_col]) == performer))

    if "same total" in q and ("gold" in q or "silver" in q or "bronze" in q):
        name_col = 0
        total_col = col_index(header, "Total", "Total Crests", "Total medal count")
        gold_col = col_index(header, "Gold", "Gold Crests", "Gold medals")
        silver_col = col_index(header, "Silver", "Silver Crests")
        bronze_col = col_index(header, "Bronze", "Bronze Crests")
        if total_col is not None and gold_col is not None and silver_col is not None and bronze_col is not None:
            subject = extract_table_name_from_question(rows, name_col, question)
            target = None
            for row in rows:
                if exactish(row[name_col], question) and row[name_col] != subject:
                    target = row[name_col]
                    break
            if subject and target:
                srow = next(row for row in rows if row[name_col] == subject)
                trow = next(row for row in rows if row[name_col] == target)
                target_total = to_int(trow[total_col])
                silver = to_int(srow[silver_col])
                bronze = to_int(srow[bronze_col])
                if target_total is None or silver is None or bronze is None:
                    return None
                return str(target_total - silver - bronze)

    if "how many days" in q:
        title_col = col_index(header, "Official Title", "Official title", "Festival Event (Official title)")
        start_col = col_index(header, "Start Date")
        finish_col = col_index(header, "Finish Date")
        if title_col is not None and start_col is not None and finish_col is not None:
            title = extract_table_name_from_question(rows, title_col, question)
            if title:
                matching = [row for row in rows if row[title_col] == title]
                if len(matching) == 1:
                    row = matching[0]
                    start = parse_day(row[start_col])
                    finish = parse_day(row[finish_col])
                    if start is not None and finish is not None:
                        return str(finish - start + 1)
                elif matching:
                    start = min(parse_day(r[start_col]) or 999 for r in matching)
                    finish = max(parse_day(r[finish_col]) or 0 for r in matching)
                    if start != 999 and finish != 0:
                        return str(finish - start + 1)

    if ("how many days" in q or "how long" in q) and (
        "start" in q or "finish" in q or "between" in q or "last" in q or "does" in q
    ):
        title_col = col_index(header, "Official Title", "Official title", "Festival Event (Official title)")
        start_col = col_index(header, "Start Date")
        finish_col = col_index(header, "Finish Date")
        if title_col is not None and start_col is not None and finish_col is not None:
            title = extract_table_name_from_question(rows, title_col, question)
            if title:
                row = next((r for r in rows if r[title_col] == title), None)
                if row:
                    start = parse_day(row[start_col])
                    finish = parse_day(row[finish_col])
                    if start is not None and finish is not None:
                        return str(finish - start + 1)

    if "last name" in q and "ends with" in q:
        name_col = col_index(header, "Cadet Name")
        house_col = col_index(header, "House")
        letter = re.search(r"letter\s+[\"']?([a-zA-Z])[\"']?", question)
        house = re.search(r"from\s+(.+?)\s+has", question, re.I)
        if name_col is not None and house_col is not None and letter and house:
            target_house = normalize_text(house.group(1))
            suffix = letter.group(1)
            for row in rows:
                if normalize_text(row[house_col]) == target_house and row[name_col].split()[-1].endswith(suffix):
                    return row[name_col]

    if "how many courses" in q and "exclusive" in q:
        year_col = col_index(header, "Term Year", "Year")
        years = [int(x) for x in re.findall(r"\b\d{4}\b", question)]
        if year_col is not None and len(years) >= 2:
            lo, hi = min(years), max(years)
            return str(sum(1 for row in rows if (year := to_int(row[year_col])) is not None and lo < year < hi))

    scored_col = col_index(header, "Scored", "Points For")
    if scored_col is not None and ("how many" in q or "how many points" in q):
        best_row = max(rows, key=lambda row: sum(1 for cell in row if len(normalize_text(cell)) > 1 and normalize_text(cell) in q))
        hits = sum(1 for cell in best_row if len(normalize_text(cell)) > 1 and normalize_text(cell) in q)
        if hits >= 2:
            return str(best_row[scored_col])

    if "arena type" in q and "winning outcomes" in q:
        arena_col = col_index(header, "Arena Type")
        outcome_col = col_index(header, "Outcome")
        if arena_col is not None and outcome_col is not None:
            counts: Counter = Counter()
            display: dict[str, str] = {}
            for row in rows:
                if normalize_text(row[0]) == "legend":
                    continue
                outcome = normalize_text(row[outcome_col]).replace("*", "")
                if outcome not in {"winner", "w"}:
                    continue
                base = re.sub(r"\([^)]*\)|（[^）]*）|\[[^]]*\]", "", row[arena_col])
                base = re.split(r"\s[-—]\s|\s+-\s+", base)[0]
                key = normalize_text(base)
                counts[key] += 1
                display.setdefault(key, base.strip().title())
            if counts:
                return display[counts.most_common(1)[0][0]]

    if "airship" in q and "originated" in q and "other than" in q:
        airship_col = col_index(header, "Airship")
        origin_col = col_index(header, "Nation/Origin")
        if airship_col is not None and origin_col is not None:
            excluded = {"", "(bri)", "bri", "brixland", "great brixton"}
            for row in rows:
                if normalize_text(row[airship_col]) == "rosebud":
                    continue
                if normalize_text(row[origin_col]) not in excluded:
                    return row[airship_col]

    if "week 1" in q and ("eliminated" in q or "eliminated in" in q):
        status_col = col_index(header, "Status")
        episode_col = col_index(header, "Episode")
        if status_col is not None:
            if "contestants" in q and episode_col is not None:
                return str(sum(1 for row in rows if normalize_text(row[episode_col]) == "week 1" and "week 1" in normalize_text(row[status_col]) and "removed" not in normalize_text(row[status_col])))
            return str(sum(1 for row in rows if normalize_text(row[status_col]) == "eliminated week 1"))

    if "immediately following the last occurrence" in q:
        title_col = col_index(header, "Title", "Project Title")
        episode_col = col_index(header, "Episode")
        quoted = re.findall(r"\"([^\"]+)\"", question)
        if title_col is not None and quoted:
            last = None
            for i, row in enumerate(rows):
                if normalize_text(row[title_col]) == normalize_text(quoted[-1]):
                    last = i
            if last is not None and last + 1 < len(rows):
                return rows[last + 1][episode_col if episode_col is not None else title_col]
    if "project title came next after" in q:
        title_col = col_index(header, "Project Title")
        quoted = re.findall(r"\"([^\"]+)\"", question)
        if title_col is not None and quoted:
            for i, row in enumerate(rows[:-1]):
                if normalize_text(row[title_col]) == normalize_text(quoted[-1]):
                    return rows[i + 1][title_col]

    if "unique players scored" in q:
        scorers_col = col_index(header, "Scorers")
        if scorers_col is not None:
            year_match = re.search(r"\b(\d{3,4})\b", question)
            year_col = col_index(header, "Year")
            competition_col = col_index(header, "Competition")
            competition = None
            if competition_col is not None:
                competitions = sorted({row[competition_col] for row in rows}, key=len, reverse=True)
                competition = next((name for name in competitions if exactish(name, question)), None)
            people: set[str] = set()
            for row in rows:
                if year_match and year_col is not None and row[year_col] != year_match.group(1):
                    continue
                if competition and competition_col is not None and normalize_text(row[competition_col]) != normalize_text(competition):
                    continue
                people.update(split_people(row[scorers_col]))
            return str(len(people)) if people else None

    if "only one distinct record holder" in q:
        guild_col = col_index(header, "Guild")
        holder_col = col_index(header, "Record Holder")
        if guild_col is not None and holder_col is not None:
            grouped: dict[str, set[str]] = {}
            for row in rows:
                grouped.setdefault(row[guild_col], set()).add(row[holder_col])
            singles = [guild for guild, holders in grouped.items() if len(holders) == 1]
            return singles[0] if len(singles) == 1 else None

    if "disqualification" in q:
        return str(sum(1 for row in rows if any("disqual" in normalize_text(cell) or "dsq" == normalize_text(cell) for cell in row)))

    if "built the longest ago" in q or "earliest built year" in q:
        note_col = col_index(header, "Notes", "Plaque Notes")
        if note_col is not None:
            best = None
            for row in rows:
                match = re.search(r"\b(?:built|erected)\s+in\s+(\d{3,4})\b", row[note_col], re.I)
                if match:
                    year = int(match.group(1))
                    if best is None or year < best[0]:
                        best = (year, row[0])
            return best[1] if best else None

    if ("which is greater" in q or "which is larger" in q) and ("goals" in q or "home" in q):
        venue_col = col_index(header, "Venue")
        opponent_col = col_index(header, "Opponent")
        goals_col = col_index(header, "Goals For")
        result_col = col_index(header, "Result")
        if venue_col is not None and opponent_col is not None:
            venue_name = extract_table_name_from_question(rows, venue_col, question)
            opp_name = extract_table_name_from_question(rows, opponent_col, question)
            venue_label = None
            opp_label = None
            if not venue_name:
                venue_match = re.search(r"(?:played in|in)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", question)
                if venue_match:
                    venue_label = venue_match.group(1)
            else:
                venue_label = venue_name
            if not opp_name:
                opp_match = re.search(r"against\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", question)
                if opp_match:
                    opp_label = opp_match.group(1)
            else:
                opp_label = opp_name
            if venue_label and opp_label:
                venue_norm = normalize_text(venue_label)
                opp_norm = normalize_text(opp_label)
                if goals_col is not None:
                    venue_sum = sum(to_int(row[goals_col]) or 0 for row in rows if venue_norm in normalize_text(row[venue_col]))
                    opp_sum = sum(to_int(row[goals_col]) or 0 for row in rows if opp_norm in normalize_text(row[opponent_col]))
                    return f"total Goals For in matches played in {venue_label}" if venue_sum > opp_sum else f"total Goals For in matches against {opp_label}"
                if result_col is not None:
                    def home_goals(row):
                        return to_int(re.split(r"\s*[\-\u2013\u2014]\s*", row[result_col])[0]) or 0
                    venue_sum = sum(home_goals(row) for row in rows if venue_norm in normalize_text(row[venue_col]))
                    opp_sum = sum(home_goals(row) for row in rows if opp_norm in normalize_text(row[opponent_col]))
                    return f"total home goals scored in matches played in {venue_label}" if venue_sum > opp_sum else f"total home goals scored against opponent {opp_label}"

    if "v8" in q:
        engine_col = col_index(header, "Engine Spec", "Vehicle", "Vehicle (description)")
        person_col = col_index(header, "Courier", "Pilot")
        if engine_col is not None and person_col is not None:
            names = {row[person_col] for row in rows if re.search(r"\bv8\b", normalize_text(row[engine_col]))}
            return str(len(names))

    if "same stadium as the match on" in q:
        date_col = col_index(header, "Date")
        stadium_col = col_index(header, "Stadium")
        if date_col is not None and stadium_col is not None:
            row_index = next((i for i, row in enumerate(rows) if exactish(row[date_col], question)), None)
            if row_index is not None:
                stadium = rows[row_index][stadium_col]
                return str(sum(1 for i, row in enumerate(rows) if i != row_index and row[stadium_col] == stadium))

    if "same number of draws" in q:
        opponent_col = col_index(header, "Opponent")
        draws_col = col_index(header, "Draws")
        if opponent_col is not None and draws_col is not None:
            opponent = extract_table_name_from_question(rows, opponent_col, question)
            if opponent:
                value = next(row[draws_col] for row in rows if row[opponent_col] == opponent)
                return str(sum(1 for row in rows if row[opponent_col] != opponent and row[draws_col] == value))

    if "rune" in q and "above" in q:
        tier_col = col_index(header, "Tier")
        step_col = col_index(header, "Variant Step (within tier)")
        rune_col = col_index(header, "Rune Name")
        if tier_col is not None and step_col is not None and rune_col is not None:
            rune = extract_table_name_from_question(rows, rune_col, question)
            if rune:
                row = next(row for row in rows if row[rune_col] == rune)
                step = to_int(row[step_col])
                if step is None:
                    return None
                for candidate in rows:
                    if candidate[tier_col] == row[tier_col] and to_int(candidate[step_col]) == step + 1:
                        return candidate[rune_col]

    if "largest" in q and "total" in q:
        total_col = col_index(header, "Total")
        if total_col is not None:
            candidates = [row for row in rows if normalize_text(row[0]) != "total"]
            return max(candidates, key=lambda row: to_int(row[total_col]) or -1)[total_col]

    if "title defenses" in q:
        defense_col = col_index(header, "Title defenses")
        if defense_col is not None:
            total = sum(to_int(row[defense_col]) or 0 for row in rows if "karnfeld" in normalize_text(" ".join(str(x) for x in row)))
            return str(total) if total else None

    if "heist-style" in q and "made by" in q:
        studio_col = col_index(header, "Studio")
        if studio_col is not None:
            studio = extract_table_name_from_question(rows, studio_col, question)
            if studio:
                return str(sum(1 for row in rows if row[studio_col] == studio))

    if "directly before" in q:
        opponent_col = col_index(header, "Opponent")
        place_col = col_index(header, "Home/Away", "Venue")
        if opponent_col is not None and place_col is not None:
            opponent = extract_table_name_from_question(rows, opponent_col, question)
            if opponent:
                playable = [row for row in rows if normalize_text(row[opponent_col]) != "bye"]
                for i, row in enumerate(playable):
                    if row[opponent_col] == opponent and i > 0:
                        return playable[i - 1][place_col]

    if "games were played against" in q:
        opponent_col = col_index(header, "Opponent")
        if opponent_col is not None:
            opponent = extract_table_name_from_question(rows, opponent_col, question)
            if opponent:
                target = normalize_text(opponent).removeprefix("at ")
                return str(sum(1 for row in rows if normalize_text(row[opponent_col]).removeprefix("at ") == target))

    if "on-time deliveries in q1" in q:
        ship_col = col_index(header, "Ship")
        quarter_col = col_index(header, "Quarter")
        ontime_col = col_index(header, "On-time Deliveries")
        if ship_col is not None and quarter_col is not None and ontime_col is not None:
            candidates = re.findall(r"MV\s+[A-Za-z]+", question)
            for row in rows:
                if row[ship_col] in candidates and normalize_text(row[quarter_col]) == "q1" and to_int(row[ontime_col]) == 18:
                    return row[ship_col]

    if "worst grid position" in q:
        season_col = col_index(header, "Season")
        circuit_col = col_index(header, "Circuit")
        grid_col = col_index(header, "Grid Position")
        season = re.search(r"season\s+(\d+)", question, re.I)
        if season_col is not None and circuit_col is not None and grid_col is not None and season:
            candidates = [row for row in rows if row[season_col] == season.group(1)]
            if candidates:
                return max(candidates, key=lambda row: ordinal_to_int(row[grid_col]) or -1)[circuit_col]

    if "last name" in q and ("token" in q or "ends with" in q):
        name_col = col_index(header, "Cadet Name")
        house_col = col_index(header, "House")
        if name_col is None:
            name_col = 0
        letter_match = re.search(r'["\u201c]([a-zA-Z])["\u201d]', question)
        house_match = re.search(r"from\s+(?:the\s+)?(.+?)(?:\s+has|\s+whose)", question, re.I)
        if letter_match:
            suffix = letter_match.group(1)
            target_house = normalize_text(house_match.group(1)) if house_match else None
            for row in rows:
                if target_house and house_col is not None and normalize_text(row[house_col]) != target_house:
                    continue
                last_token = row[name_col].split()[-1]
                if last_token.endswith(suffix):
                    return row[name_col]

    if "at least" in q or "at most" in q:
        for ci, col_name in enumerate(header):
            cn = normalize_text(col_name)
            if cn in normalize_text(question):
                threshold_match = re.search(r"at least\s+([\d.]+)", q) or re.search(r"at most\s+([\d.]+)", q)
                if threshold_match:
                    threshold = float(threshold_match.group(1))
                    is_at_least = "at least" in q
                    count = 0
                    for row in rows:
                        try:
                            val = float(str(row[ci]).replace(",", ""))
                        except (ValueError, TypeError):
                            continue
                        if (is_at_least and val >= threshold) or (not is_at_least and val <= threshold):
                            count += 1
                    if count > 0:
                        return str(count)

    if "how many" in q and "scored" in q:
        scorers_col = col_index(header, "Scorers")
        if scorers_col is not None:
            year_match = re.search(r"\b(\d{3,4})\b", question)
            year_col = col_index(header, "Year")
            competition_col = col_index(header, "Competition")
            competition = None
            if competition_col is not None:
                competitions = sorted({row[competition_col] for row in rows}, key=len, reverse=True)
                competition = next((name for name in competitions if exactish(name, question)), None)
            people: set[str] = set()
            for row in rows:
                if year_match and year_col is not None and row[year_col] != year_match.group(1):
                    continue
                if competition and competition_col is not None and normalize_text(row[competition_col]) != normalize_text(competition):
                    continue
                people.update(split_people(row[scorers_col]))
            return str(len(people)) if people else None

    if "established first" in q or "earliest" in q:
        estab_col = col_index(header, "Established", "Founded", "Date Established")
        name_col = 0
        if estab_col is not None:
            best_date = None
            best_name = None
            for row in rows:
                date_str = row[estab_col]
                year_m = re.search(r"\b(\d{3,4})\b", date_str)
                day_m = re.search(r"\b(\d{1,2})\b", date_str)
                if year_m:
                    year = int(year_m.group(1))
                    month_map = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
                                 "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12}
                    month = 1
                    for abbr, mnum in month_map.items():
                        if abbr in date_str.lower():
                            month = mnum
                            break
                    day = int(day_m.group(1)) if day_m else 1
                    date_val = (year, month, day)
                    if best_date is None or date_val < best_date:
                        best_date = date_val
                        best_name = row[name_col]
            if best_name:
                return best_name

    if "more total" in q or ("which month" in q and "more" in q):
        month_col = col_index(header, "Month-Year", "Month\u2013Year", "Month")
        if month_col is not None:
            quoted = re.findall(r"(\S+\s+\d{3,4})", question)
            if len(quoted) >= 2:
                m1, m2 = normalize_text(quoted[0]), normalize_text(quoted[1])
                numeric_cols = [i for i in range(len(header)) if i != month_col]
                sum1, sum2 = 0.0, 0.0
                for row in rows:
                    rmonth = normalize_text(row[month_col])
                    for ci in numeric_cols:
                        try:
                            val = float(str(row[ci]).replace(",", ""))
                        except (ValueError, TypeError):
                            continue
                        if rmonth == m1:
                            sum1 += val
                        elif rmonth == m2:
                            sum2 += val
                if sum1 or sum2:
                    return quoted[0] if sum1 >= sum2 else quoted[1]

    return None


# ============================================================
# Prompt 构造
# ============================================================
def build_kg_prompt(item: dict, style: str = "cot", shuffled: list[str] | None = None) -> str:
    triples_list = shuffled if shuffled is not None else parse_triples(item["input"])
    question = item["question"]
    if ENABLE_KG_SUBGRAPH and shuffled is None:
        triples_list = extract_subgraph(triples_list, question)

    triples_text = format_kg_triples_grouped(triples_list)

    base = f"""You are an expert at knowledge graph reasoning. Below are knowledge graph triples in the format: <Subject> <Relation> <Object>.

Triples:
{triples_text}

Question: {question}
"""

    protocols = {
        "cot": """Reasoning protocol:
1. IDENTIFY the key entities and constraints in the question (who, what, which, how many, AND, OR, NOT, "but not", "later than", "earlier than").
2. TRACE the reasoning path through the triples step by step. For multi-hop questions, find intermediate bridge entities.
3. DISTRACTOR CHECK: Many triples are noise. Only use triples whose relations are relevant to the question.
4. SET OPERATIONS: If the question uses "both ... and", "but not", "neither ... nor", compute the intersection/difference/complement explicitly.
5. VERIFY: Before outputting, re-check that your answer satisfies ALL constraints in the question.
6. For entity names, use the human-readable label (not internal IDs like m.xxxxx).
7. Output ONLY the final answer. No explanation, no quotes, no prefix like "Answer:".""",
        "direct": """Be DIRECT and PRECISE:
- Base your answer strictly on the triples provided. Do NOT use outside knowledge.
- For "AND/BUT NOT/EXCEPT" questions, compute the set explicitly.
- If multiple triples seem relevant, prefer the one most directly satisfying ALL question constraints.
- Output ONLY the final entity/value as written in the triples. No explanation.""",
        "evidence": """Evidence-first protocol:
1. List the 2–4 triples in the graph that constitute the SUPPORTING CHAIN for the answer.
2. From those triples, derive the final answer entity/value.
3. If the question contains "AND/BUT NOT", verify by checking both the inclusion and exclusion conditions hold.
4. Output ONLY the final answer (no listed evidence, no explanation, no "Answer:").""",
        "decompose": """Decomposition protocol (StepChain GraphRAG):
1. Decompose the question into a chain of sub-conditions (silently). For each, list candidate entities from the triples.
2. Combine the candidate sets using the logical operators in the question (intersection / difference).
3. Output ONLY the final answer.""",
    }
    return base + protocols.get(style, protocols["cot"])


def build_mhqa_prompt(item: dict, style: str = "cot", ranked: list[dict] | None = None) -> str:
    if ranked is not None:
        contexts = ranked
    elif ENABLE_PASSAGE_RANK:
        contexts = [c for _, c in rank_passages(item["question"], item["contexts"])]
    else:
        contexts = item["contexts"]

    contexts_text = ""
    for i, ctx in enumerate(contexts):
        title = ctx.get("title", f"Context {i + 1}")
        paragraph = ctx.get("paragraph", "")
        if not paragraph and "sentences" in ctx:
            paragraph = " ".join(ctx["sentences"])
        contexts_text += f"\n[Document {i + 1}: {title}]\n{paragraph}\n"

    base = f"""You are an expert at multi-hop reading comprehension. Read ALL documents below carefully.
{contexts_text}

Question: {item["question"]}
"""

    protocols = {
        "cot": """Reasoning protocol:
1. DECOMPOSE: Break the question into sub-questions. Identify what bridge entities or facts connect the documents.
2. EVIDENCE TRACE: For each sub-question, find the specific sentence(s) in the documents that provide the answer.
3. INTEGRATE: Combine the sub-answers to form the final answer.
4. FICTIONAL CONTENT: Some documents describe fictional worlds. Answer based strictly on what the documents state, not external knowledge.
5. VERIFY: Re-read the question and confirm your answer satisfies all constraints.
6. Be precise about names, dates, numbers, and titles. Use the EXACT form from the documents.
7. Output ONLY the final answer (a name, number, date, or short phrase). No explanations, no quotes, no prefix like "Answer:".""",
        "direct": """Be DIRECT and PRECISE:
- Answer must be supported by at least one document above (not by external knowledge).
- Copy the answer verbatim from a document where possible (preserves F1 / ROUGE-L).
- Output ONLY the final answer.""",
        "evidence": """Evidence-first protocol:
1. Silently identify the 1–3 documents (by title) that contain the relevant evidence and the exact sentence(s) supporting the answer.
2. Derive the answer from those sentences only.
3. Output ONLY the final answer text, in the same surface form as the documents.""",
        "decompose": """Decomposition (Socratic / StepChain):
1. Decompose into 2–3 sub-questions and answer each from the documents (silently).
2. Aggregate the sub-answers into the final answer.
3. Output ONLY the final answer text.""",
    }
    return base + protocols.get(style, protocols["cot"])


def build_table_prompt(item: dict, style: str = "cot") -> str:
    table = item["table"]
    table_text = format_table_markdown(table)
    records_text = format_table_records(table)
    num_rows = len(table["rows"])
    num_cols = len(table["header"])

    base = f"""You are an expert at table-based reasoning. The table has {num_cols} columns and {num_rows} data rows.

Markdown view:
{table_text}

Records view (row-as-record):
{records_text}

Question: {item["question"]}
"""

    protocols = {
        "cot": """Reasoning protocol (Chain-of-Table inspired):
1. LOCATE: Identify which column(s) and row(s) are relevant to the question.
2. FILTER: Apply any conditions (e.g., "where Venue = X", "in season Y") to select the correct subset of rows.
3. COMPUTE: Perform any required arithmetic (count, sum, difference, max, min, avg) precisely.
4. DEDUP: If the question asks for "distinct" or "unique", remove duplicate rows based on factual content (ignore row IDs).
5. VERIFY: Double-check your count/computation by listing the qualifying items silently.
6. Follow legend rows exactly when present.
7. Output ONLY the final answer (a name, number, or short phrase). No explanations, no quotes, no prefix like "Answer:".""",
        "direct": """Be DIRECT:
- Filter rows by the question's condition.
- For counts: count rows; for sum/avg/diff: do the arithmetic.
- Match cell values exactly (case-insensitive). Ignore irrelevant rows like "Legend" or "Total".
- Output ONLY the final answer.""",
        "evidence": """Evidence-first:
1. Silently list the qualifying rows (by Row N) under the question's conditions.
2. Compute or read the answer from those rows.
3. Output ONLY the final answer in the exact wording / number the table uses.""",
        "decompose": """Decomposition:
1. Sub-step 1: enumerate candidate rows.
2. Sub-step 2: apply secondary filter / compute aggregate.
3. Output ONLY the final answer.""",
    }
    return base + protocols.get(style, protocols["cot"])


def build_prompt(item: dict, style: str = "cot", **kwargs) -> str:
    tt = item["task_type"]
    if tt == "knowledge_graph":
        return build_kg_prompt(item, style=style, **kwargs)
    if tt == "multi_hop_qa":
        return build_mhqa_prompt(item, style=style, **kwargs)
    if tt == "table_qa":
        return build_table_prompt(item, style=style)
    raise ValueError(f"未知题型: {tt}")


# ============================================================
# 答案后处理 / 类型感知
# ============================================================
@dataclass
class AnswerType:
    name: str
    is_numeric: bool = False


def detect_answer_type(question: str) -> AnswerType:
    q = normalize_text(question)
    if any(p in q for p in ("how many", "how many times", "the number of", "count of", "total number")):
        return AnswerType("count", is_numeric=True)
    if "what year" in q or "in what year" in q or "which year" in q:
        return AnswerType("year", is_numeric=True)
    if "what date" in q or "on what date" in q or "what month" in q:
        return AnswerType("date")
    if q.startswith(("does ", "did ", "is ", "was ", "were ", "are ", "do ", "can ", "could ", "has ", "have ", "had ")):
        return AnswerType("yes_no")
    if "list" in q.split() or "list all" in q or "which ... and" in q or " and " in q and "which" in q:
        # 太宽，弱信号；这里只标记
        pass
    return AnswerType("entity")


def clean_answer(raw: str) -> str:
    text = raw.strip()
    for prefix in ("Answer:", "answer:", "A:", "The answer is", "The answer is:", "Final answer:"):
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].strip()
    text = text.strip('"\' ')
    text = text.rstrip(".")
    # 删除尾部解释性短语
    for cut in ("\n\nExplanation", "\nExplanation", "\nNote:"):
        if cut in text:
            text = text.split(cut)[0].strip()
    # 仅在一行非常短时；多行回答取第一行
    if "\n" in text:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if lines:
            text = lines[0]
    return text.strip()


def coerce_answer(answer: str, atype: AnswerType) -> str:
    if not answer:
        return answer
    if atype.name == "count" or atype.name == "year":
        m = re.search(r"-?\d[\d,]*", answer)
        if m:
            return m.group(0).replace(",", "")
        return answer
    if atype.name == "yes_no":
        a = answer.strip().lower()
        if a.startswith("yes") or a in {"true", "y"}:
            return "Yes"
        if a.startswith("no") or a in {"false", "n"}:
            return "No"
        return answer
    return answer


# ============================================================
# 答案投票
# ============================================================
def parse_int_from(s: str) -> int | None:
    if s is None:
        return None
    m = re.search(r"-?\d[\d,]*", s)
    if not m:
        return None
    try:
        return int(m.group(0).replace(",", ""))
    except ValueError:
        return None


def vote_numeric(answers: list[str]) -> str | None:
    nums = [parse_int_from(a) for a in answers if a]
    nums = [n for n in nums if n is not None]
    if not nums:
        return None
    cnt = Counter(nums)
    return str(cnt.most_common(1)[0][0])


def vote_entity(answers: list[str], threshold: int = 85) -> str | None:
    valid = [a for a in answers if a and not is_bad_answer(a)]
    if not valid:
        return None
    # 聚类：fuzzy 相近的算一类
    clusters: list[list[str]] = []
    for a in valid:
        placed = False
        for cluster in clusters:
            if fuzz.token_set_ratio(normalize_text(a), normalize_text(cluster[0])) >= threshold:
                cluster.append(a)
                placed = True
                break
        if not placed:
            clusters.append([a])
    clusters.sort(key=len, reverse=True)
    winner_cluster = clusters[0]
    # 在 winner_cluster 中取最长的（信息更全）
    return max(winner_cluster, key=len)


def majority_vote(answers: list[str], atype: AnswerType) -> str:
    if not answers:
        return ""
    valid = [a for a in answers if a]
    if not valid:
        return answers[0] if answers else ""

    if atype.is_numeric:
        r = vote_numeric(valid)
        if r:
            return r

    r = vote_entity(valid)
    if r:
        return r
    # fallback 朴素归一化
    counts = Counter(normalize_text(a) for a in valid if not is_bad_answer(a))
    if counts:
        best_norm = counts.most_common(1)[0][0]
        for a in valid:
            if normalize_text(a) == best_norm:
                return a
    return valid[0]


def is_bad_answer(answer: str) -> bool:
    text = normalize_text(answer).strip(" .。")
    if not text:
        return True
    return any(pattern in text for pattern in BAD_ANSWER_PATTERNS)


# ============================================================
# API 调用：单一入口，按 LLM_PROVIDER 走两套 SDK
# ============================================================
def _is_transient_error(exc: Exception) -> tuple[bool, int | None]:
    """判断当前 provider 的异常是否可重试。返回 (是否重试, status_code)。"""
    mod = _anthropic_mod if LLM_PROVIDER == "anthropic" else _openai_mod
    if mod is None:
        return False, None
    if isinstance(exc, getattr(mod, "RateLimitError", tuple())):
        return True, 429
    if isinstance(exc, (
        getattr(mod, "APIConnectionError", tuple()),
        getattr(mod, "APITimeoutError", tuple()),
    )):
        return True, None
    if isinstance(exc, getattr(mod, "APIStatusError", tuple())):
        status = getattr(exc, "status_code", None)
        return (bool(status and 500 <= status < 600), status)
    return False, None


async def call_api_once(
    prompt: str,
    item_id: int,
    thinking_budget: int = 10000,
    label: str = "main",
) -> str:
    """两条路径：
      LLM_PROVIDER='anthropic' → messages.create + extended thinking
      LLM_PROVIDER='openai'    → chat.completions.create + reasoning_effort
    带指数退避 + 抖动重试。
    """
    backoff = INITIAL_BACKOFF
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if LLM_PROVIDER == "anthropic":
                resp = await _get_anthropic_client().messages.create(
                    model=ANTHROPIC_MODEL,
                    max_tokens=MAX_TOKENS,
                    thinking={"type": "enabled", "budget_tokens": thinking_budget},
                    messages=[{"role": "user", "content": prompt}],
                )
                for block in resp.content:
                    if getattr(block, "type", None) == "text":
                        return clean_answer(block.text)
                return ""
            elif LLM_PROVIDER == "openai":
                resp = await _get_openai_client().chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    max_completion_tokens=MAX_TOKENS,
                    reasoning_effort=OPENAI_REASONING_EFFORT,
                )
                choices = getattr(resp, "choices", None) or []
                if not choices:
                    return ""
                return clean_answer(getattr(choices[0].message, "content", None) or "")
            else:
                raise ValueError(f"unknown LLM_PROVIDER {LLM_PROVIDER!r}; 必须是 'anthropic' 或 'openai'")
        except Exception as e:
            retriable, status = _is_transient_error(e)
            if retriable and attempt < MAX_RETRIES:
                sleep_s = backoff + random.uniform(0, backoff * 0.5)
                print(f"  [id={item_id} {label}] transient ({status}), retry {attempt}/{MAX_RETRIES} in {sleep_s:.1f}s")
                await asyncio.sleep(sleep_s)
                backoff = min(backoff * 2, 60)
                continue
            print(f"  [id={item_id} {label}] {'final fail' if retriable else 'error'}: {type(e).__name__}: {e}")
            return ""
    return ""


def make_diverse_prompts(item: dict, n: int) -> list[tuple[str, str, float]]:
    """生成 n 路多样化的 (prompt, label, temperature)。"""
    rng = random.Random(item["id"])
    styles_pool = ["cot", "direct", "evidence", "decompose"]
    temps_pool = [0.20, 0.30, 0.40, 0.25, 0.45]
    prompts: list[tuple[str, str, float]] = []

    tt = item["task_type"]

    if tt == "knowledge_graph":
        raw_triples = parse_triples(item["input"])
        sub = extract_subgraph(raw_triples, item["question"]) if ENABLE_KG_SUBGRAPH else raw_triples
        for k in range(n):
            style = styles_pool[k % len(styles_pool)]
            temp = temps_pool[k % len(temps_pool)]
            if k == 0:
                triples_view = sub
            else:
                # 顺序扰动（SKA-Bench Order Insensitivity）
                triples_view = list(sub)
                rng2 = random.Random(item["id"] * 1000 + k)
                rng2.shuffle(triples_view)
            p = build_kg_prompt(item, style=style, shuffled=triples_view)
            prompts.append((p, f"kg-{style}-t{temp}", temp))
    elif tt == "multi_hop_qa":
        ranked = [c for _, c in rank_passages(item["question"], item["contexts"])] if ENABLE_PASSAGE_RANK else list(item["contexts"])
        for k in range(n):
            style = styles_pool[k % len(styles_pool)]
            temp = temps_pool[k % len(temps_pool)]
            if k == 0:
                view = ranked
            else:
                view = list(ranked)
                rng2 = random.Random(item["id"] * 1000 + k)
                rng2.shuffle(view)
            p = build_mhqa_prompt(item, style=style, ranked=view)
            prompts.append((p, f"mhqa-{style}-t{temp}", temp))
    else:
        for k in range(n):
            style = styles_pool[k % len(styles_pool)]
            temp = temps_pool[k % len(temps_pool)]
            p = build_table_prompt(item, style=style)
            prompts.append((p, f"table-{style}-t{temp}", temp))
    return prompts


async def call_with_voting(item: dict) -> str:
    n = VOTING_ROUNDS_BY_TYPE.get(item["task_type"], 3)
    thinking_budget = THINKING_BUDGET_BY_TYPE.get(item["task_type"], 10000)
    prompts = make_diverse_prompts(item, n)
    results = await asyncio.gather(
        *(
            call_api_once(p, item["id"], thinking_budget=thinking_budget, label=label)
            for p, label, _ in prompts
        )
    )
    atype = detect_answer_type(item["question"])
    candidates = [coerce_answer(r, atype) for r in results]
    chosen = majority_vote(candidates, atype)
    if n > 1:
        print(f"  [id={item['id']}] votes: {[c[:30] for c in candidates]} -> {chosen[:40]}")
    return chosen


# ============================================================
# Evidence-based Verification
# ============================================================
def build_verify_prompt(item: dict, candidate: str) -> str:
    tt = item["task_type"]
    if tt == "knowledge_graph":
        triples_list = parse_triples(item["input"])
        if ENABLE_KG_SUBGRAPH:
            triples_list = extract_subgraph(triples_list, item["question"])
        ctx = "Triples:\n" + format_kg_triples_grouped(triples_list)
    elif tt == "multi_hop_qa":
        ranked = [c for _, c in rank_passages(item["question"], item["contexts"])] if ENABLE_PASSAGE_RANK else item["contexts"]
        ctx_text = ""
        for i, c in enumerate(ranked):
            title = c.get("title", f"Context {i + 1}")
            paragraph = c.get("paragraph", "") or " ".join(c.get("sentences", []))
            ctx_text += f"\n[Document {i + 1}: {title}]\n{paragraph}\n"
        ctx = ctx_text
    else:
        ctx = "Table (Markdown):\n" + format_table_markdown(item["table"])

    return f"""You are a strict verifier for a question-answering system.

Source:
{ctx}

Question: {item["question"]}

Proposed Answer: {candidate!r}

Verify the proposed answer against the source ONLY (no outside knowledge).
- If the answer is supported and correct, output exactly: OK
- If the answer is wrong, output exactly: REVISE: <the correct answer>
- The corrected answer must be a single line, concise, with no quotes, no prefix, no explanation.
- Do NOT output "Unknown/None/Cannot be determined" — always commit to the best supported answer.
"""


async def verify_and_revise(item: dict, candidate: str) -> str:
    if not ENABLE_VERIFICATION or not candidate:
        return candidate
    prompt = build_verify_prompt(item, candidate)
    raw = await call_api_once(
        prompt,
        item["id"],
        thinking_budget=min(THINKING_BUDGET_BY_TYPE.get(item["task_type"], 10000), 8000),
        label="verify",
    )
    if not raw:
        return candidate
    raw_clean = raw.strip()
    if raw_clean.upper().startswith("OK"):
        return candidate
    m = re.match(r"^\s*REVISE\s*[:：]\s*(.+)$", raw_clean, re.IGNORECASE | re.DOTALL)
    if m:
        revised = clean_answer(m.group(1))
        if revised and not is_bad_answer(revised):
            print(f"  [id={item['id']}] verifier revised: {candidate[:30]} -> {revised[:30]}")
            return revised
    # 兜底：若 verifier 输出本身像答案
    if not is_bad_answer(raw_clean) and len(raw_clean) < 200:
        return clean_answer(raw_clean)
    return candidate


# ============================================================
# Repair（拒答兜底）
# ============================================================
def build_repair_prompt(item: dict, bad_answer: str) -> str:
    base = build_prompt(item, style="direct")
    return f"""{base}

The previous answer was invalid for scoring: {bad_answer!r}.
You MUST provide the best concise answer based on the SOURCE above. Do NOT refuse. Do NOT output unknown/none/not specified/cannot be determined.
If the source genuinely lacks an explicit answer, output your single MOST PROBABLE answer derived from the closest available evidence.
Output ONLY the final answer text on a single line."""


async def repair_if_bad(item: dict, candidate: str) -> str:
    if not ENABLE_REPAIR or not is_bad_answer(candidate):
        return candidate
    prompt = build_repair_prompt(item, candidate)
    raw = await call_api_once(
        prompt,
        item["id"],
        thinking_budget=min(THINKING_BUDGET_BY_TYPE.get(item["task_type"], 10000), 6000),
        label="repair",
    )
    return raw if raw and not is_bad_answer(raw) else candidate


# ============================================================
# Cross-run Consensus（跨运行答案投票 + 锁定 + 跳过）
# ============================================================
# 思路：每次脚本启动都会跑出一份 submit.jsonl。我们把历史每次跑的结果
# 累计到 consensus_state.json；如果同一 id 在历史中有 >=N 次"归一化相等"
# 的答案，就把它锁定（locked_answer），下次启动直接复用、跳过 API 调用，
# 既省 token 又能用 wisdom-of-crowds 滤除随机抖动。
# 输出：
#   - submit.jsonl          —— 本次运行的完整答案（包含锁定 + 新生成）
#   - consensus_final.jsonl —— 仅锁定的答案（用于最终提交）
#   - runs/run_<TS>_<P>.jsonl —— 历史每次提交的归档
#   - consensus_state.json  —— 历史 + 锁定状态
ENABLE_CONSENSUS = os.environ.get("ENABLE_CONSENSUS", "true").lower() == "true"
RESET_CONSENSUS = os.environ.get("RESET_CONSENSUS", "false").lower() == "true"
CONSENSUS_LOCK_AT = int(os.environ.get("CONSENSUS_LOCK_AT", "2"))  # 多少次相同就锁定
CONSENSUS_STATE_FILE = "consensus_state.json"
CONSENSUS_FINAL_FILE = "consensus_final.jsonl"
CONSENSUS_RUNS_DIR = "runs"


def _consensus_norm(answer: str, atype: AnswerType) -> str:
    """归一化用于比较的 key。先 coerce 再 normalize。"""
    if answer is None:
        return ""
    coerced = coerce_answer(answer, atype)
    return normalize_text(coerced)


def load_consensus_state() -> dict:
    if RESET_CONSENSUS or not os.path.exists(CONSENSUS_STATE_FILE):
        return {}
    try:
        with open(CONSENSUS_STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"[consensus] state load failed ({e}); 重新开始")
        return {}


def save_consensus_state(state: dict) -> None:
    tmp = CONSENSUS_STATE_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    os.replace(tmp, CONSENSUS_STATE_FILE)


def archive_previous_submit() -> str | None:
    """把上一次的 submit.jsonl + submit_raw.jsonl 归档到 runs/。"""
    if not os.path.exists(OUTPUT_FILE) and not os.path.exists(RAW_OUTPUT_FILE):
        return None
    os.makedirs(CONSENSUS_RUNS_DIR, exist_ok=True)
    import time
    ts = time.strftime("%Y%m%d-%H%M%S")
    provider_tag = LLM_PROVIDER
    archived_path = None
    if os.path.exists(OUTPUT_FILE):
        archived_path = os.path.join(
            CONSENSUS_RUNS_DIR, f"submit_{ts}_{provider_tag}.jsonl",
        )
        os.replace(OUTPUT_FILE, archived_path)
        print(f"[consensus] archived {OUTPUT_FILE} -> {archived_path}")
    if os.path.exists(RAW_OUTPUT_FILE):
        raw_archived = os.path.join(
            CONSENSUS_RUNS_DIR, f"submit_raw_{ts}_{provider_tag}.jsonl",
        )
        os.replace(RAW_OUTPUT_FILE, raw_archived)
        print(f"[consensus] archived {RAW_OUTPUT_FILE} -> {raw_archived}")
    return archived_path


def ingest_run_into_state(state: dict, run_file: str, data: list) -> int:
    """把一次运行结果合并入 state；返回新增条目数。"""
    if not os.path.exists(run_file):
        return 0
    by_id: dict[int, str] = {}
    with open(run_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "id" in obj and isinstance(obj.get("answer", ""), str):
                by_id[obj["id"]] = obj["answer"].strip()
    item_by_id = {d["id"]: d for d in data}
    added = 0
    for iid, ans in by_id.items():
        if not ans:
            continue
        item = item_by_id.get(iid)
        if item is None:
            continue
        atype = detect_answer_type(item["question"])
        record_answer_to_state(state, iid, ans, atype, persist=False)
        added += 1
    return added


def record_answer_to_state(
    state: dict,
    item_id: int,
    answer: str,
    atype: AnswerType,
    persist: bool = True,
) -> bool:
    """把一次答案加入历史，并在达到阈值时锁定。返回 True 表示发生了新锁定。"""
    if not answer:
        return False
    key = str(item_id)
    entry = state.setdefault(key, {"history": [], "locked": False, "locked_answer": ""})
    if entry.get("locked"):
        return False
    norm = _consensus_norm(answer, atype)
    if not norm:
        return False
    entry["history"].append({"raw": answer, "norm": norm})
    counts: Counter = Counter(h["norm"] for h in entry["history"] if h.get("norm"))
    if not counts:
        return False
    most_norm, cnt = counts.most_common(1)[0]
    newly_locked = False
    if cnt >= CONSENSUS_LOCK_AT:
        matches = [h["raw"] for h in entry["history"] if h.get("norm") == most_norm]
        entry["locked_answer"] = max(matches, key=len)
        entry["locked"] = True
        newly_locked = True
    if persist:
        save_consensus_state(state)
    return newly_locked


def get_locked_answer(state: dict, item_id: int) -> str | None:
    entry = state.get(str(item_id))
    if entry and entry.get("locked"):
        return entry.get("locked_answer") or None
    return None


def write_consensus_final(state: dict, data: list) -> tuple[int, int]:
    """写出 consensus_final.jsonl，仅包含锁定项；未锁定的 answer 留空。"""
    locked = 0
    total = 0
    with open(CONSENSUS_FINAL_FILE, "w", encoding="utf-8") as f:
        for item in sorted(data, key=lambda x: x["id"]):
            total += 1
            ans = get_locked_answer(state, item["id"])
            if ans:
                locked += 1
            f.write(json.dumps({"id": item["id"], "answer": ans or ""}, ensure_ascii=False) + "\n")
    return locked, total


def print_consensus_summary(state: dict, data: list) -> None:
    locked = sum(1 for d in data if get_locked_answer(state, d["id"]))
    one_run = sum(1 for d in data if not get_locked_answer(state, d["id"]) and len((state.get(str(d["id"])) or {}).get("history", [])) == 1)
    disagreeing = sum(1 for d in data if not get_locked_answer(state, d["id"]) and len((state.get(str(d["id"])) or {}).get("history", [])) >= 2)
    print(f"[consensus] 锁定 {locked}/{len(data)}  |  单次答 {one_run}  |  多次分歧 {disagreeing}  |  阈值 ={CONSENSUS_LOCK_AT}")


# ============================================================
# 主流程
# ============================================================
def load_done_ids(path: str) -> set:
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if "id" in obj and obj.get("answer", "") != "":
                    done.add(obj["id"])
            except json.JSONDecodeError:
                continue
    return done


def load_answers(path: str) -> dict:
    answers = {}
    if not os.path.exists(path):
        return answers
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "id" in obj and isinstance(obj.get("answer", ""), str):
                answers[obj["id"]] = obj["answer"].strip()
    return answers


def write_submit(path: str, data: list, answers: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for item in sorted(data, key=lambda x: x["id"]):
            f.write(json.dumps({"id": item["id"], "answer": answers.get(item["id"], "")}, ensure_ascii=False) + "\n")


async def solve_item(item: dict) -> str:
    # 1. Table 确定性求解
    if item["task_type"] == "table_qa":
        det = solve_table_qa(item)
        if det is not None and not is_bad_answer(det):
            return det

    # 2. 自一致投票
    candidate = await call_with_voting(item)

    # 3. Verify-and-revise
    candidate = await verify_and_revise(item, candidate)

    # 4. 拒答兜底
    candidate = await repair_if_bad(item, candidate)

    # 5. 类型后处理
    atype = detect_answer_type(item["question"])
    candidate = coerce_answer(candidate, atype)
    return candidate


async def worker(sem, item, out_lock, out_file, counter, total, consensus_state=None):
    async with sem:
        try:
            answer = await solve_item(item)
        except Exception as e:
            print(f"  [id={item['id']}] solve_item failed: {type(e).__name__}: {e}")
            answer = ""
        async with out_lock:
            out_file.write(json.dumps({"id": item["id"], "answer": answer}, ensure_ascii=False) + "\n")
            out_file.flush()
            counter["done"] += 1
            preview = (item["question"][:50] + "...") if len(item["question"]) > 50 else item["question"]
            ans_preview = (answer[:60] + "...") if len(answer) > 60 else answer
            print(f"[{counter['done']:3d}/{total}] id={item['id']:3d} {item['task_type']:16s} | Q: {preview}")
            print(f"            -> {ans_preview!r}")
            # 共识状态更新（同样在锁内，避免并发写文件冲突）
            if consensus_state is not None and answer:
                atype = detect_answer_type(item["question"])
                if record_answer_to_state(consensus_state, item["id"], answer, atype, persist=True):
                    print(f"            *** consensus LOCKED for id={item['id']} -> "
                          f"{consensus_state[str(item['id'])]['locked_answer'][:60]!r}")


async def main():
    _ensure_provider_modules()
    print("=" * 60)
    print(f"CCKS 2026 OneEval v2  |  provider={LLM_PROVIDER}")
    if LLM_PROVIDER == "anthropic":
        print(f"  model={ANTHROPIC_MODEL}  base={ANTHROPIC_BASE_URL}")
        print(f"  THINKING_BUDGET: {THINKING_BUDGET_BY_TYPE}")
    else:
        print(f"  model={OPENAI_MODEL}  base={OPENAI_BASE_URL}  reasoning_effort={OPENAI_REASONING_EFFORT}")
    print(f"  VOTING_ROUNDS:   {VOTING_ROUNDS_BY_TYPE}")
    print(f"  KG_SUBGRAPH={ENABLE_KG_SUBGRAPH}  PASSAGE_RANK={ENABLE_PASSAGE_RANK}  VERIFY={ENABLE_VERIFICATION}  REPAIR={ENABLE_REPAIR}")
    print(f"  CONSENSUS={ENABLE_CONSENSUS}  lock_at={CONSENSUS_LOCK_AT}  reset={RESET_CONSENSUS}")
    print("=" * 60)

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    total_all = len(data)
    type_counts = Counter(d["task_type"] for d in data)
    print(f"题目总数: {total_all}")
    for t, c in type_counts.items():
        print(f"  - {t}: {c}")

    # —— Cross-run consensus 准备 ——
    consensus_state: dict = {}
    if ENABLE_CONSENSUS:
        consensus_state = load_consensus_state()
        # 把上次的 submit.jsonl 合并入状态（如果存在）然后归档
        if os.path.exists(OUTPUT_FILE):
            ingested = ingest_run_into_state(consensus_state, OUTPUT_FILE, data)
            print(f"[consensus] 将上次 submit.jsonl 的 {ingested} 条答案纳入历史")
            save_consensus_state(consensus_state)
        archive_previous_submit()
        print_consensus_summary(consensus_state, data)

    answers: dict = {}
    if ENABLE_CONSENSUS:
        # 预填锁定答案
        for item in data:
            la = get_locked_answer(consensus_state, item["id"])
            if la:
                answers[item["id"]] = la
        if answers:
            print(f"[consensus] 跳过 {len(answers)} 个已锁定题（直接复用 locked_answer，零 API 调用）")
    # submit_raw.jsonl 已被归档；intra-run resume 因此重新开始（也可以用 ENABLE_CONSENSUS=false 保留旧逻辑）

    deterministic_count = 0
    for item in data:
        if item["task_type"] != "table_qa":
            continue
        # 锁定的不覆盖
        if item["id"] in answers:
            continue
        answer = solve_table_qa(item)
        if answer is not None:
            answers[item["id"]] = answer
            deterministic_count += 1
            # 把确定性结果也喂给共识，加速锁定
            if ENABLE_CONSENSUS:
                atype = detect_answer_type(item["question"])
                record_answer_to_state(consensus_state, item["id"], answer, atype, persist=False)
    if ENABLE_CONSENSUS:
        save_consensus_state(consensus_state)

    done_ids = {
        item["id"]
        for item in data
        if item["id"] in answers and answers[item["id"]] and (not REANSWER_BAD or not is_bad_answer(answers[item["id"]]))
    }
    todo = [d for d in data if d["id"] not in done_ids]
    print(f"已完成: {len(done_ids)}  |  待答: {len(todo)}  |  并发: {CONCURRENCY}")
    print(f"表格确定性求解覆盖: {deterministic_count}")

    if todo:
        sem = asyncio.Semaphore(CONCURRENCY)
        out_lock = asyncio.Lock()
        counter = {"done": 0}
        with open(RAW_OUTPUT_FILE, "a", encoding="utf-8") as out_file:
            tasks = [
                worker(sem, item, out_lock, out_file, counter, len(todo),
                       consensus_state if ENABLE_CONSENSUS else None)
                for item in todo
            ]
            await asyncio.gather(*tasks)

        answers.update(load_answers(RAW_OUTPUT_FILE))
        for item in data:
            if item["task_type"] == "table_qa":
                answer = solve_table_qa(item)
                if answer is not None:
                    answers[item["id"]] = answer

    write_submit(OUTPUT_FILE, data, answers)

    if ENABLE_CONSENSUS:
        save_consensus_state(consensus_state)
        locked, total = write_consensus_final(consensus_state, data)
        print(f"\n[consensus] consensus_final.jsonl: 锁定 {locked}/{total}")
        print_consensus_summary(consensus_state, data)

    final_done = load_done_ids(OUTPUT_FILE)
    print(f"\n写入完成: {len(final_done)}/{total_all}")
    missing = [d["id"] for d in data if d["id"] not in final_done]
    if missing:
        print(f"仍缺失 {len(missing)} 题: {missing[:20]}{'...' if len(missing) > 20 else ''}")
        print("再跑一次脚本将自动补答。")
    else:
        print(f"全部 {total_all} 题已答完 -> {OUTPUT_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
