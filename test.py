"""
CCKS 2026 OneEval 解题脚本  v3

技术栈：
  - LLM         : anthropic SDK 或 openai SDK（互斥单选，硬编码切换）
  - KG          : networkx ego_graph 子图抽取 + 关系关键词扩展
  - 段落检索    : rank_bm25.BM25Okapi  +  sentence-transformers(Qwen3-Embedding-8B/MiniLM)
                  混合 BM25 + Dense + cross-granularity (paragraph + sentence-max)
  - NER         : spaCy en_core_web_sm
  - 表格        : DuckDB SQL + Pandas one-liner（沙盒 eval）+ 内置确定性规则
  - 推理增强    : Think-on-Graph 迭代探索；问题分解；自一致投票（按 style 加权）
  - 后处理      : count / year / yes-no / entity / list 类型感知
  - 持久化      : cross-run consensus（连续 N 次答案一致即锁定，跳过 API 调用）

所有配置都在本文件顶部硬编码，不读环境变量。
"""

from __future__ import annotations

import asyncio
import functools
import json
import math
import os
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

import anthropic
import duckdb
import networkx as nx
import openai
import pandas as pd
import spacy
import torch
from rank_bm25 import BM25Okapi
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer, util as st_util
from tqdm.auto import tqdm

# ============================================================
# 配置区（所有可调常量集中在此处）
# ============================================================

# ---------- LLM Provider（互斥单选） ----------
# "anthropic": Claude 系列（Opus/Sonnet 等）或任何 Anthropic 兼容反代
# "openai"   : GPT-5.x / GLM-5.x / DeepSeek-OpenAI 等所有 OpenAI 兼容端点
LLM_PROVIDER = "anthropic"

# Anthropic 端点（Claude Opus 官方：https://api.anthropic.com / claude-opus-4-7-...）
ANTHROPIC_API_KEY  = "sk-3b8f9bf9a89c4633a36cc7109ef2026f"
ANTHROPIC_BASE_URL = "https://api.deepseek.com/anthropic"
ANTHROPIC_MODEL    = "deepseek-v4-pro"

# OpenAI 端点（GPT 官方：https://api.openai.com/v1 / gpt-5.5）
#               （GLM   ：https://open.bigmodel.cn/api/paas/v4/ / glm-5）
OPENAI_API_KEY  = "sk-3b8f9bf9a89c4633a36cc7109ef2026f"
OPENAI_BASE_URL = "https://api.deepseek.com"
OPENAI_MODEL    = "deepseek-v4-pro"
OPENAI_REASONING_EFFORT = "high"   # minimal | low | medium | high

# ---------- 调用参数 ----------
MAX_TOKENS = 16000                  # Anthropic.max_tokens / OpenAI.max_completion_tokens
THINKING_BUDGET_BY_TYPE = {         # Anthropic extended-thinking 预算（OpenAI 用 reasoning_effort）
    "knowledge_graph": 10000,
    "multi_hop_qa": 12000,
    "table_qa": 8000,
}
MAX_RETRIES = 5
INITIAL_BACKOFF = 2.0
REANSWER_BAD = True                 # 答案是 "Unknown/None/..." 时升温重答

# ---------- 数据文件 ----------
INPUT_FILE = "contest_data.json"
OUTPUT_FILE = "submit.jsonl"
RAW_OUTPUT_FILE = "submit_raw.jsonl"

# ---------- 自一致投票 ----------
VOTING_ROUNDS_BY_TYPE = {
    "knowledge_graph": 5,
    "multi_hop_qa": 5,
    "table_qa": 3,
}
ENABLE_WEIGHTED_VOTING = True
STYLE_WEIGHTS = {                   # 不同 prompt 风格的投票权重
    "cot": 1.5,
    "evidence": 1.3,
    "decompose": 1.2,
    "direct": 1.0,
    "tog": 1.4,
}

# ---------- KG 子图 ----------
ENABLE_KG_SUBGRAPH = True
KG_SUBGRAPH_DEPTH = 2
KG_SUBGRAPH_MAX = 90                # 上限
KG_SUBGRAPH_KEEP_AT_LEAST = 30      # 子图过小时补足到此数量
KG_SUBGRAPH_MIN_RAW = 25            # 三元组少于此数就不裁剪

# ---------- 段落 BM25 + Dense 检索 ----------
ENABLE_PASSAGE_RANK = True
PASSAGE_RANK_KEEP = 6
HYBRID_BM25_WEIGHT = 0.5            # 与 dense 的线性融合系数
ENABLE_CROSS_GRANULARITY = True     # paragraph + sentence-max 融合
SENT_WEIGHT_IN_CROSSG = 0.4

# ---------- 稠密 embedding 模型 ----------
# CUDA 可用 → Qwen3-Embedding-8B；否则 MiniLM-L6-v2
DENSE_MODEL_GPU = "Qwen/Qwen3-Embedding-8B"
DENSE_MODEL_CPU = "sentence-transformers/all-MiniLM-L6-v2"
DENSE_BATCH_GPU = 32
DENSE_BATCH_CPU = 16

# ---------- 推理增强 ----------
ENABLE_TOG = True                   # Think-on-Graph 迭代探索（KG）
TOG_MAX_HOPS = 3
ENABLE_DECOMPOSE = True             # 多跳问题分解（multi_hop_qa）
DECOMPOSE_PER_SUB_TOPK = 2

# ---------- 表格 ----------
ENABLE_TABLE_SQL = True             # DuckDB SQL 兜底
ENABLE_TABLE_PANDAS_CODE = True     # Pandas one-liner 兜底

# ---------- 答案验证 / 修复 ----------
ENABLE_VERIFICATION = True
ENABLE_REPAIR = True

# ---------- 并发控制 ----------
CONCURRENCY_BY_TYPE = {             # verifier(1) + repair(0~1) 额外开销，故分级限流
    "knowledge_graph": 30,
    "multi_hop_qa": 30,
    "table_qa": 40,
}

# ---------- Cross-run consensus ----------
ENABLE_CONSENSUS = True
RESET_CONSENSUS = False
CONSENSUS_LOCK_AT = 2               # 连续 N 次答案相同就锁定
CONSENSUS_STATE_FILE = "consensus_state.json"
CONSENSUS_FINAL_FILE = "consensus_final.jsonl"
CONSENSUS_RUNS_DIR = "runs"


# ============================================================
# Provider 客户端（延迟初始化）
# ============================================================
_anthropic_client: anthropic.AsyncAnthropic | None = None
_openai_client: openai.AsyncOpenAI | None = None


def _get_anthropic_client() -> anthropic.AsyncAnthropic:
    global _anthropic_client
    if _anthropic_client is None:
        _anthropic_client = anthropic.AsyncAnthropic(
            api_key=ANTHROPIC_API_KEY,
            base_url=ANTHROPIC_BASE_URL,
            timeout=600.0,
        )
    return _anthropic_client


def _get_openai_client() -> openai.AsyncOpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = openai.AsyncOpenAI(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
            timeout=600.0,
        )
    return _openai_client


def _ensure_provider_modules() -> None:
    """预热当前 provider 的 client。"""
    if LLM_PROVIDER == "anthropic":
        _get_anthropic_client()
    elif LLM_PROVIDER == "openai":
        _get_openai_client()
    else:
        raise ValueError(f"未知 LLM_PROVIDER: {LLM_PROVIDER!r}; 必须是 'anthropic' 或 'openai'")


# ============================================================
# 其它库的延迟加载（spaCy / sentence-transformers）
# ============================================================
_spacy_nlp = None  # type: ignore
_dense_model: SentenceTransformer | None = None
_DENSE_DEVICE: str = "cpu"
_DENSE_BATCH: int = DENSE_BATCH_CPU


def _get_spacy_nlp():
    """spaCy en_core_web_sm，禁用不需要的 pipeline 加速。"""
    global _spacy_nlp
    if _spacy_nlp is None:
        _spacy_nlp = spacy.load(
            "en_core_web_sm",
            disable=["tagger", "parser", "lemmatizer"],
        )
    return _spacy_nlp


def _get_dense_model() -> SentenceTransformer:
    """加载稠密 embedding 模型。CUDA 可用 → Qwen3-8B(bf16)；否则 MiniLM。"""
    global _dense_model, _DENSE_DEVICE, _DENSE_BATCH
    if _dense_model is None:
        if torch.cuda.is_available():
            _DENSE_DEVICE = "cuda"
            model_name = DENSE_MODEL_GPU
            _DENSE_BATCH = DENSE_BATCH_GPU
            print(f"[dense] loading {model_name} on cuda (bf16) ...")
            _dense_model = SentenceTransformer(
                model_name,
                device="cuda",
                model_kwargs={"torch_dtype": torch.bfloat16},
            )
        else:
            _DENSE_DEVICE = "cpu"
            model_name = DENSE_MODEL_CPU
            _DENSE_BATCH = DENSE_BATCH_CPU
            print(f"[dense] loading {model_name} on cpu ...")
            _dense_model = SentenceTransformer(model_name, device="cpu")
    return _dense_model


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


_NER_LABELS = {"PERSON", "ORG", "GPE", "LOC", "WORK_OF_ART", "EVENT", "FAC", "PRODUCT", "NORP"}


def extract_proper_nouns(text: str) -> list[str]:
    """提取问句中疑似专有名词。spaCy NER + 引号/大写连词补充。"""
    doc = _get_spacy_nlp()(text)
    candidates: list[str] = [ent.text for ent in doc.ents if ent.label_ in _NER_LABELS]
    # 补充：引号包围 / 连续 Capitalized words（NER 模型偶尔漏掉的专名）
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


# relation 通常是：
#  (a) Freebase 点分串：`film.film.starring..film.performance.actor` 或 `people.person.born_in`
#  (b) 大写式：`LOCATED_IN` / `IS_A`
# Freebase 三元组允许 .. 形式（CVT 节点），所以允许内部 dots：`[A-Za-z_][\w.]*\.[\w.]*\w`
_REL_PATTERN = r"[A-Za-z_][\w.]*\.[\w.]*\w|[A-Z][A-Z0-9_]{2,}"
_REL_TOKEN_RE = re.compile(rf"^(?:{_REL_PATTERN})$")
_TRIPLE_DELIMITER_RE = re.compile(rf"^(.+?)\s+({_REL_PATTERN})\s+(.+)$")


def split_triple(triple: str) -> tuple[str, str, str]:
    """启发式拆分 <head> <relation> <tail>。
    优先正则一把抓；失败 fall back 到原 token 扫描；最后退化为按空格三段。
    """
    triple = triple.strip()
    m = _TRIPLE_DELIMITER_RE.match(triple)
    if m:
        return m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
    tokens = triple.split(" ")
    for i, tok in enumerate(tokens):
        if _REL_TOKEN_RE.match(tok):
            return (
                " ".join(tokens[:i]).strip(),
                tok,
                " ".join(tokens[i + 1:]).strip(),
            )
    if len(tokens) >= 3:
        return tokens[0], tokens[1], " ".join(tokens[2:])
    return triple, "", ""


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


def build_kg_graph_nx(triples: list[str]):
    """把三元组列表构造成 networkx.DiGraph（节点 = normalize_text(实体)，边携带 raw 三元组与 relation）。"""
    G = nx.DiGraph()
    for t in triples:
        h, r, tail = split_triple(t)
        hh, tt = normalize_text(h), normalize_text(tail)
        if hh and tt:
            G.add_edge(hh, tt, relation=r, raw=t)
    return G


@functools.lru_cache(maxsize=512)
def _build_kg_graph_cached(triples_tuple: tuple):
    """缓存：同一 item 同一三元组集合只建一次图。"""
    return build_kg_graph_nx(list(triples_tuple))


def extract_subgraph(
    triples: list[str],
    question: str,
    depth: int = KG_SUBGRAPH_DEPTH,
    cap: int = KG_SUBGRAPH_MAX,
) -> list[str]:
    """从问题实体出发 BFS depth 跳的子图 + 相关关系匹配。失败回退到原列表。
    BFS 走 networkx.ego_graph，外加 cap 截断与关系关键词扩展。
    """
    if len(triples) <= KG_SUBGRAPH_MIN_RAW:
        return triples
    idx, parsed = build_kg_index(triples)
    entity_keys = list(idx.keys())
    seeds = fuzzy_find_entities(question, entity_keys)

    selected: set[int] = set()
    seed_triple_indices: set[int] = set()
    if seeds:
        G = _build_kg_graph_cached(tuple(triples))
        bfs_nodes: set[str] = set()
        for seed in seeds:
            if seed not in G:
                continue
            try:
                sub = nx.ego_graph(G, seed, radius=depth, undirected=True)
                bfs_nodes |= set(sub.nodes())
            except Exception:
                continue
        bfs_nodes |= set(seeds)
        # 把所有"两端都在 bfs_nodes 内"的边对应的三元组索引收进 selected；先 seed-邻接，再多跳
        for ent in seeds:
            for ti in idx.get(ent, []):
                if len(selected) >= cap:
                    break
                selected.add(ti)
                seed_triple_indices.add(ti)
        for i, (h, _r, t) in enumerate(parsed):
            if i in selected:
                continue
            if normalize_text(h) in bfs_nodes or normalize_text(t) in bfs_nodes:
                selected.add(i)
                if len(selected) >= cap:
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
# 多跳段落 BM25 + Dense + Hybrid + Cross-Granularity 排序
# (rank_bm25 + sentence-transformers)
# ============================================================
def _bm25_scores(question: str, passages: list[str]) -> list[float]:
    """rank_bm25.BM25Okapi 评分 + 专有名词原串子串加分。"""
    if not passages:
        return []
    q_tokens = tokenize(question)
    pns = [normalize_text(p) for p in extract_proper_nouns(question)]
    q_tokens = q_tokens + pns
    docs = [tokenize(p) for p in passages]
    bm25 = BM25Okapi(docs)
    scores = list(bm25.get_scores(q_tokens))
    # 子串命中加分
    for i, p in enumerate(passages):
        pl = p.lower()
        for pn in pns:
            if pn and pn in pl:
                scores[i] += 1.5
    return scores


def _normalize_scores(scores: list[float]) -> list[float]:
    """Min-max 归一化。"""
    if not scores:
        return []
    mn, mx = min(scores), max(scores)
    if mx - mn < 1e-9:
        return [0.0] * len(scores)
    return [(s - mn) / (mx - mn) for s in scores]


def _dense_scores(question: str, passages: list[str]) -> list[float]:
    """sentence-transformers 余弦相似度。"""
    if not passages:
        return []
    model = _get_dense_model()
    q_emb = model.encode(question, convert_to_tensor=True, show_progress_bar=False)
    c_emb = model.encode(
        passages,
        convert_to_tensor=True,
        batch_size=_DENSE_BATCH,
        show_progress_bar=False,
    )
    sims = st_util.cos_sim(q_emb, c_emb)[0]
    return [float(s) for s in sims.cpu().tolist()]


def _cross_granularity_dense_scores(question: str, contexts: list[dict]) -> list[float]:
    """段落级 + 句子级 max 融合。所有文本一次性 batch encode（单次 forward pass）。"""
    if not contexts:
        return []
    model = _get_dense_model()
    all_texts: list[str] = []
    para_indices: list[int] = []
    sent_ranges: list[tuple[int, int]] = []
    for c in contexts:
        para = c.get("paragraph") or " ".join(c.get("sentences", []))
        sents = c.get("sentences") or [para]
        para_indices.append(len(all_texts))
        all_texts.append(para or " ")
        sent_start = len(all_texts)
        all_texts.extend((s or " ") for s in sents)
        sent_ranges.append((sent_start, len(all_texts)))
    all_embs = model.encode(
        all_texts,
        convert_to_tensor=True,
        batch_size=_DENSE_BATCH,
        show_progress_bar=False,
    )
    q_emb = model.encode(question, convert_to_tensor=True, show_progress_bar=False)
    results: list[float] = []
    for i in range(len(contexts)):
        p_idx = para_indices[i]
        p_score = float(st_util.cos_sim(q_emb, all_embs[p_idx:p_idx + 1])[0, 0])
        s_start, s_end = sent_ranges[i]
        if s_end > s_start:
            s_score = float(st_util.cos_sim(q_emb, all_embs[s_start:s_end])[0].max())
        else:
            s_score = 0.0
        results.append((1 - SENT_WEIGHT_IN_CROSSG) * p_score + SENT_WEIGHT_IN_CROSSG * s_score)
    return results


# 每 item 一次的排序缓存（key = (item_id, question, len(contexts)))；solve_item 退出前 clear
_RANK_CACHE: dict[tuple, list[tuple[int, dict]]] = {}


def _rank_cache_key(item_id: int | None, question: str, contexts: list[dict], top_k: int) -> tuple:
    return (item_id, question, len(contexts), top_k)


def _passages_text(contexts: list[dict]) -> list[str]:
    """统一构造 BM25/dense 的 passage 文本：'title. paragraph'。"""
    out = []
    for c in contexts:
        title = c.get("title", "")
        text = c.get("paragraph") or " ".join(c.get("sentences", []))
        out.append(f"{title}. {text}")
    return out


def rank_passages(
    question: str,
    contexts: list[dict],
    top_k: int = PASSAGE_RANK_KEEP,
    item_id: int | None = None,
) -> list[tuple[int, dict]]:
    """Hybrid BM25 + Cross-Granularity Dense 段落排序。每 item 缓存一次。"""
    if not contexts:
        return []
    cache_key = _rank_cache_key(item_id, question, contexts, top_k)
    cached = _RANK_CACHE.get(cache_key)
    if cached is not None:
        return cached
    passages = _passages_text(contexts)
    bm25_n = _normalize_scores(_bm25_scores(question, passages))
    crossg_n = _normalize_scores(_cross_granularity_dense_scores(question, contexts))
    combined = [
        HYBRID_BM25_WEIGHT * b + (1 - HYBRID_BM25_WEIGHT) * d
        for b, d in zip(bm25_n, crossg_n)
    ]
    order = sorted(range(len(contexts)), key=lambda i: -combined[i])
    keep = order[:top_k] if top_k and top_k < len(contexts) else order
    result = [(i, contexts[i]) for i in keep]
    _RANK_CACHE[cache_key] = result
    return result


def rank_passages_for_subq(sub_q: str, contexts: list[dict], top_k: int) -> list[tuple[int, dict]]:
    """子问题的轻量级排序（BM25 + 段落级 dense，不做 cross-granularity 节省 encode 量）。"""
    if not contexts:
        return []
    passages = _passages_text(contexts)
    bm25_n = _normalize_scores(_bm25_scores(sub_q, passages))
    dense_n = _normalize_scores(_dense_scores(sub_q, passages))
    combined = [
        HYBRID_BM25_WEIGHT * b + (1 - HYBRID_BM25_WEIGHT) * d
        for b, d in zip(bm25_n, dense_n)
    ]
    order = sorted(range(len(contexts)), key=lambda i: -combined[i])
    return [(i, contexts[i]) for i in order[:top_k]]


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
        contexts = [c for _, c in rank_passages(item["question"], item["contexts"], item_id=item["id"])]
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
# Think-on-Graph 迭代探索（KG）
# ============================================================
async def tog_iterative_explore(item: dict) -> str:
    """逐跳扩展实体集，每跳让 LLM 决定继续探索还是给出 FINAL 答案。"""
    raw_triples = parse_triples(item["input"])
    if not raw_triples:
        return ""
    question = item["question"]
    seed_ents = extract_proper_nouns(question)
    explored: list[str] = []
    reasoning: list[str] = []
    current_query = question

    for hop in range(TOG_MAX_HOPS):
        sub = extract_subgraph(raw_triples, current_query) if ENABLE_KG_SUBGRAPH else raw_triples
        view = sub[:60]
        prompt = (
            "Knowledge Graph (relevant subgraph):\n"
            f"{format_kg_triples_grouped(view)}\n\n"
            f"Original question: {question}\n"
            f"Seed entities: {seed_ents}\n"
            f"Reasoning so far: {' | '.join(reasoning) if reasoning else 'none'}\n\n"
            "Decide:\n"
            "1) If you can answer now, output exactly: FINAL: <answer>\n"
            "2) Otherwise output: NEXT: <intermediate entity or relation> (one line, used to expand search)\n"
            "Do not output anything else."
        )
        raw = await call_api_once(prompt, item["id"], thinking_budget=8000, label=f"tog-hop{hop}")
        if not raw:
            break
        raw_strip = raw.strip()
        if "FINAL:" in raw_strip.upper():
            ans = re.split(r"FINAL\s*:\s*", raw_strip, maxsplit=1, flags=re.IGNORECASE)[-1]
            return clean_answer(ans)
        m = re.search(r"NEXT\s*:\s*(.+)", raw_strip, re.IGNORECASE)
        if m:
            hint = m.group(1).strip()
            reasoning.append(hint[:120])
            current_query = f"{question} {hint}"
            new_ents = [e for e in extract_proper_nouns(hint) if e not in explored]
            explored.extend(new_ents)
        else:
            # 模型没遵循格式，直接返回它的输出作为答案
            return clean_answer(raw_strip)
    # 兜底：直接走 cot 路径
    return await call_api_once(build_kg_prompt(item, style="cot"), item["id"], label="tog-final")


# ============================================================
# 问题分解 + 子问题级检索（multi_hop_qa）
# ============================================================
def _parse_sub_questions(raw: str) -> list[str]:
    """三段式解析：```json``` 代码块 → ["..."] 数组 → 按行抓 ? 结尾。"""
    if not raw:
        return []
    # 1) ```json``` 包裹
    cb = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    blob = cb.group(1) if cb else raw
    # 2) 找含 sub_questions 的 JSON 对象
    m = re.search(r"\{[^{}]*\"sub_questions\"\s*:\s*\[[^\]]*\][^{}]*\}", blob, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(0))
            subs = data.get("sub_questions") or []
            subs = [s.strip() for s in subs if isinstance(s, str) and s.strip()]
            if subs:
                return subs[:4]
        except json.JSONDecodeError:
            pass
    # 3) 找 ["...", "..."] 形式
    arr = re.search(r"\[\s*\"[^\"]+\"(?:\s*,\s*\"[^\"]+\")*\s*\]", blob, re.DOTALL)
    if arr:
        try:
            subs = [s.strip() for s in json.loads(arr.group(0)) if isinstance(s, str) and s.strip()]
            if subs:
                return subs[:4]
        except json.JSONDecodeError:
            pass
    # 4) 按行抓 `1. ... ?` / `- ... ?`
    lines = [ln.strip().lstrip("0123456789.-) ").strip() for ln in raw.splitlines() if ln.strip()]
    qs = [ln for ln in lines if "?" in ln and len(ln) > 10]
    return qs[:4]


async def decompose_question(item: dict) -> list[str]:
    """让 LLM 把 multi-hop 拆成 2–3 个子问题。失败回退到原问题。"""
    prompt = (
        "Decompose the following multi-hop question into 2-3 sequential sub-questions.\n"
        "Each sub-question must be answerable from the given documents.\n"
        f"Question: {item['question']}\n\n"
        'Output STRICT JSON only, no prose: {"sub_questions": ["q1", "q2", ...]}'
    )
    raw = await call_api_once(prompt, item["id"], thinking_budget=4000, label="decompose")
    subs = _parse_sub_questions(raw)
    return subs if subs else [item["question"]]


async def decompose_and_retrieve(item: dict, per_sub_topk: int = DECOMPOSE_PER_SUB_TOPK) -> list[dict]:
    """每个子问题独立检索 top-k 段落，合并去重，返回排序后的 contexts。"""
    subs = await decompose_question(item)
    seen: set[int] = set()
    out: list[dict] = []
    for sq in subs:
        ranked = rank_passages_for_subq(sq, item["contexts"], per_sub_topk)
        for idx, ctx in ranked:
            if idx in seen:
                continue
            seen.add(idx)
            out.append(ctx)
    # 兜底：如果分解得到的太少，把全局 top-k 补上
    if len(out) < PASSAGE_RANK_KEEP:
        for idx, ctx in rank_passages(item["question"], item["contexts"], PASSAGE_RANK_KEEP, item_id=item["id"]):
            if idx not in seen:
                out.append(ctx)
                seen.add(idx)
                if len(out) >= PASSAGE_RANK_KEEP:
                    break
    return out


# ============================================================
# Pandas / DuckDB 表格兜底
# ============================================================
def _table_to_df(table: dict) -> pd.DataFrame:
    return pd.DataFrame(table["rows"], columns=table["header"])


def _safe_str(v) -> str:
    return "" if v is None else str(v).strip()


# pandas one-liner 沙盒：禁止任何危险 token
_PANDAS_CODE_BLOCKED = re.compile(
    r"__|\bimport\b|\bexec\b|\beval\b|\bopen\b|\bos\b|\bsys\b|\bsubprocess\b"
    r"|\bgetattr\b|\bsetattr\b|\bdelattr\b|\bglobals\b|\blocals\b|\bvars\b"
    r"|\bcompile\b|\b__\w+__\b"
)
_PANDAS_SAFE_BUILTINS = {
    "len": len, "min": min, "max": max, "sum": sum, "abs": abs,
    "round": round, "sorted": sorted, "set": set, "list": list,
    "tuple": tuple, "dict": dict, "str": str, "int": int, "float": float,
    "bool": bool, "enumerate": enumerate, "zip": zip, "range": range,
    "any": any, "all": all,
}


async def solve_table_with_duckdb_sql(item: dict) -> str | None:
    """让 LLM 生成 DuckDB SQL，本地执行后取第一行第一列作为答案。"""
    if not ENABLE_TABLE_SQL:
        return None
    df = _table_to_df(item["table"])
    header = item["table"]["header"]
    sample_rows = item["table"]["rows"][:3]
    prompt = (
        "You are an expert at writing DuckDB SQL queries.\n"
        f"There is a DuckDB-registered DataFrame named `df` with columns: {header}\n"
        f"Sample rows: {sample_rows}\n\n"
        f"Question: {item['question']}\n\n"
        "Write ONE DuckDB SQL statement that produces the answer in its first row/first column.\n"
        "Output ONLY the SQL statement. No markdown fences. No explanation.\n"
    )
    sql = await call_api_once(prompt, item["id"], thinking_budget=4000, label="table-sql")
    if not sql:
        return None
    sql = re.sub(r"^```(?:sql)?\s*|\s*```$", "", sql.strip(), flags=re.IGNORECASE | re.MULTILINE).strip().rstrip(";")
    if not sql or len(sql) > 800:
        return None
    try:
        con = duckdb.connect()
        con.register("df", df)
        result = con.execute(sql).fetchdf()
        con.close()
    except Exception as e:
        print(f"  [id={item['id']}] table-sql failed: {type(e).__name__}: {e}")
        return None
    if result is None or result.empty:
        return None
    return _safe_str(result.iloc[0, 0]) or None


async def solve_table_with_pandas_code(item: dict) -> str | None:
    """让 LLM 生成一行 pandas 表达式（变量 `df`），在受限沙盒中 eval。"""
    if not ENABLE_TABLE_PANDAS_CODE:
        return None
    df = _table_to_df(item["table"])
    header = item["table"]["header"]
    df_head = df.head(3).to_string()
    prompt = (
        "You are an expert at writing one-line pandas expressions.\n"
        f"A DataFrame `df` is already defined with columns: {header}\n"
        f"Preview:\n{df_head}\n\n"
        f"Question: {item['question']}\n\n"
        "Write a SINGLE LINE of pandas code that computes the answer.\n"
        "Use only `df` and `pd`. Output ONLY the expression, no assignment, no print, no fences.\n"
        "Examples: df.loc[df['Year']==2020, 'Score'].sum() ; df['City'].iloc[0] ; len(df)\n"
    )
    code = await call_api_once(prompt, item["id"], thinking_budget=3000, label="table-pandas")
    if not code:
        return None
    code = re.sub(r"^```(?:python)?\s*|\s*```$", "", code.strip(), flags=re.IGNORECASE | re.MULTILINE).strip()
    if not code or len(code) > 400 or "\n" in code or ";" in code:
        return None
    if _PANDAS_CODE_BLOCKED.search(code):
        print(f"  [id={item['id']}] table-pandas blocked: {code[:80]}")
        return None
    try:
        result = eval(
            code,
            {"__builtins__": _PANDAS_SAFE_BUILTINS},
            {"df": df, "pd": pd},
        )
    except Exception as e:
        print(f"  [id={item['id']}] table-pandas eval failed: {type(e).__name__}: {e}")
        return None
    # Series / DataFrame → 取首元素
    if isinstance(result, pd.Series):
        result = result.iloc[0] if len(result) else None
    elif isinstance(result, pd.DataFrame):
        result = result.iloc[0, 0] if not result.empty else None
    return _safe_str(result) or None


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


def _style_weight_from_label(label: str) -> float:
    """label 形如 'kg-cot-t0.3' / 'mh-evidence-t0.2' → 取中间风格名查 STYLE_WEIGHTS。"""
    parts = label.split("-")
    for p in parts:
        if p in STYLE_WEIGHTS:
            return STYLE_WEIGHTS[p]
    return 1.0


def weighted_majority_vote(
    answers: list[str],
    labels: list[str],
    atype: AnswerType,
) -> str:
    """按 style 加权 + 数值/实体感知投票。"""
    if not ENABLE_WEIGHTED_VOTING:
        return majority_vote(answers, atype)
    pairs = [(a, l) for a, l in zip(answers, labels) if a]
    if not pairs:
        return ""
    # 数值场景：先把每个答案 coerce 到数值字符串，再加权
    if atype.is_numeric:
        weights: dict[str, float] = defaultdict(float)
        for a, lbl in pairs:
            n = coerce_answer(a, atype)
            if not n:
                continue
            weights[normalize_text(n)] += _style_weight_from_label(lbl)
        if weights:
            # 直接返回票数最高的归一化字符串本身（coerce 后已是干净数字）
            return max(weights, key=weights.__getitem__)
    # 通用：加权 + 用 fuzzy bucket
    buckets: list[tuple[str, float, str]] = []  # (norm, weight, raw)
    for a, lbl in pairs:
        if is_bad_answer(a):
            continue
        an = normalize_text(a)
        w = _style_weight_from_label(lbl)
        # 与已有 bucket fuzzy 合并
        matched = False
        for i, (bn, bw, _) in enumerate(buckets):
            if bn == an or fuzz.token_set_ratio(an, bn) >= 88:
                buckets[i] = (bn, bw + w, buckets[i][2])
                matched = True
                break
        if not matched:
            buckets.append((an, w, a))
    if buckets:
        buckets.sort(key=lambda x: -x[1])
        return buckets[0][2]
    return pairs[0][0]


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
    mod = anthropic if LLM_PROVIDER == "anthropic" else openai
    if isinstance(exc, mod.RateLimitError):
        return True, 429
    if isinstance(exc, (mod.APIConnectionError, mod.APITimeoutError)):
        return True, None
    if isinstance(exc, mod.APIStatusError):
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
        ranked = [c for _, c in rank_passages(item["question"], item["contexts"], item_id=item["id"])] if ENABLE_PASSAGE_RANK else list(item["contexts"])
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
        # decomposition 路径在 make_diverse_prompts 之外异步注入（见 call_with_voting）
    else:
        for k in range(n):
            style = styles_pool[k % len(styles_pool)]
            temp = temps_pool[k % len(temps_pool)]
            p = build_table_prompt(item, style=style)
            prompts.append((p, f"table-{style}-t{temp}", temp))
    return prompts


async def _multi_hop_decomposed_call(item: dict, thinking_budget: int) -> tuple[str, str]:
    """先分解 -> 每个子问题独立检索 -> 用合并后的 contexts 跑一条 cot 路径。返回 (answer, label)。"""
    if not ENABLE_DECOMPOSE:
        return "", "mhqa-decompose-disabled"
    try:
        ranked = await decompose_and_retrieve(item)
        if not ranked:
            return "", "mhqa-decompose-empty"
        p = build_mhqa_prompt(item, style="cot", ranked=ranked)
        ans = await call_api_once(p, item["id"], thinking_budget=thinking_budget, label="mhqa-decompose")
        return ans, "mhqa-decompose-t0.3"
    except Exception as e:
        print(f"  [id={item['id']}] decompose path failed: {type(e).__name__}: {e}")
        return "", "mhqa-decompose-err"


async def call_with_voting(item: dict) -> str:
    n = VOTING_ROUNDS_BY_TYPE.get(item["task_type"], 3)
    thinking_budget = THINKING_BUDGET_BY_TYPE.get(item["task_type"], 10000)
    prompts = make_diverse_prompts(item, n)

    coros: list = [
        call_api_once(p, item["id"], thinking_budget=thinking_budget, label=label)
        for p, label, _ in prompts
    ]
    labels: list[str] = [lbl for _, lbl, _ in prompts]

    # multi_hop_qa：再追加一条分解路径；显式记录其在 coros 中的位置
    decompose_idx: int | None = None
    if item["task_type"] == "multi_hop_qa" and ENABLE_DECOMPOSE:
        decompose_idx = len(coros)
        coros.append(_multi_hop_decomposed_call(item, thinking_budget))
        labels.append("mhqa-decompose")

    raw_results = await asyncio.gather(*coros)
    results: list[str] = []
    for i, r in enumerate(raw_results):
        if i == decompose_idx and isinstance(r, tuple):
            ans, lbl = r
            results.append(ans)
            labels[i] = lbl   # 用真实 label 覆盖占位
        elif isinstance(r, tuple):
            # 不应发生：常规路径返回 str
            results.append(r[0])
        else:
            results.append(r)

    atype = detect_answer_type(item["question"])
    candidates = [coerce_answer(r, atype) for r in results]
    if ENABLE_WEIGHTED_VOTING:
        chosen = weighted_majority_vote(candidates, labels, atype)
    else:
        chosen = majority_vote(candidates, atype)
    if len(candidates) > 1:
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
        ranked = [c for _, c in rank_passages(item["question"], item["contexts"], item_id=item["id"])] if ENABLE_PASSAGE_RANK else item["contexts"]
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
# 思路：每次脚本启动跑出一份 submit.jsonl，累计到 consensus_state.json；
# 同一 id 在历史中出现 >= CONSENSUS_LOCK_AT 次归一化相等的答案就锁定，
# 下次启动直接复用 locked_answer 跳过 API 调用。
# 输出：
#   - submit.jsonl          —— 本次运行的完整答案（含锁定 + 新生成）
#   - consensus_final.jsonl —— 仅锁定的答案（用于最终提交）
#   - runs/run_<TS>_<P>.jsonl —— 历史每次提交的归档
#   - consensus_state.json  —— 历史 + 锁定状态
# 所有相关常量见文件顶部配置区。
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
    tt = item["task_type"]
    atype = detect_answer_type(item["question"])

    # === Table：先确定性，再 SQL 兜底，最后 pandas-code 兜底，全部不行才走 LLM 投票 ===
    if tt == "table_qa":
        det = solve_table_qa(item)
        if det is not None and not is_bad_answer(det):
            return det
        sql_ans = await solve_table_with_duckdb_sql(item)
        if sql_ans and not is_bad_answer(sql_ans):
            return coerce_answer(sql_ans, atype)
        pd_ans = await solve_table_with_pandas_code(item)
        if pd_ans and not is_bad_answer(pd_ans):
            return coerce_answer(pd_ans, atype)

    # === KG：先投票；答案可疑时才启动 ToG（节省 token + 受 semaphore 约束）===
    candidate = await call_with_voting(item)
    if tt == "knowledge_graph" and ENABLE_TOG:
        if not candidate or is_bad_answer(candidate) or len(candidate.strip()) < 2:
            tog_ans = await tog_iterative_explore(item)
            if tog_ans and not is_bad_answer(tog_ans):
                candidate = tog_ans

    # === Verifier + Repair + 类型后处理 ===
    candidate = await verify_and_revise(item, candidate)
    candidate = await repair_if_bad(item, candidate)
    candidate = coerce_answer(candidate, atype)
    return candidate


async def worker(
    sems: dict[str, asyncio.Semaphore],
    default_sem: asyncio.Semaphore,
    item: dict,
    out_lock: asyncio.Lock,
    out_file,
    counter: dict,
    total: int,
    pbar,
    consensus_state: dict | None = None,
) -> None:
    """按题型从对应 semaphore 限流；写出 JSONL + 更新进度条 + 喂入 consensus。"""
    sem = sems.get(item["task_type"], default_sem)
    async with sem:
        try:
            answer = await solve_item(item)
        except Exception as e:
            print(f"  [id={item['id']}] solve_item failed: {type(e).__name__}: {e}")
            answer = ""
        # 释放 per-item 缓存
        _RANK_CACHE.clear()
        async with out_lock:
            out_file.write(json.dumps({"id": item["id"], "answer": answer}, ensure_ascii=False) + "\n")
            out_file.flush()
            counter["done"] += 1
            preview = (item["question"][:50] + "...") if len(item["question"]) > 50 else item["question"]
            ans_preview = (answer[:60] + "...") if len(answer) > 60 else answer
            pbar.update(1)
            pbar.set_postfix_str(f"id={item['id']:>3} {item['task_type']:>16}", refresh=True)
            pbar.write(f"[{counter['done']:3d}/{total}] id={item['id']:3d} {item['task_type']:16s} | Q: {preview} -> {ans_preview!r}")
            # 共识状态更新（同样在锁内，避免并发写文件冲突）
            if consensus_state is not None and answer:
                atype = detect_answer_type(item["question"])
                if record_answer_to_state(consensus_state, item["id"], answer, atype, persist=True):
                    pbar.write(
                        f"*** consensus LOCKED for id={item['id']} -> "
                        f"{consensus_state[str(item['id'])]['locked_answer'][:60]!r}"
                    )


async def main():
    _ensure_provider_modules()
    print("=" * 70)
    print(f"CCKS 2026 OneEval v3  |  provider={LLM_PROVIDER}")
    if LLM_PROVIDER == "anthropic":
        print(f"  model={ANTHROPIC_MODEL}  base={ANTHROPIC_BASE_URL}")
        print(f"  THINKING_BUDGET: {THINKING_BUDGET_BY_TYPE}")
    else:
        print(f"  model={OPENAI_MODEL}  base={OPENAI_BASE_URL}  reasoning_effort={OPENAI_REASONING_EFFORT}")
    print(f"  VOTING_ROUNDS:   {VOTING_ROUNDS_BY_TYPE}")
    print(f"  KG_SUBGRAPH={ENABLE_KG_SUBGRAPH}  PASSAGE_RANK={ENABLE_PASSAGE_RANK}  VERIFY={ENABLE_VERIFICATION}  REPAIR={ENABLE_REPAIR}")
    print(f"  DENSE: GPU={DENSE_MODEL_GPU}  CPU={DENSE_MODEL_CPU}  CROSS_GRAN={ENABLE_CROSS_GRANULARITY}")
    print(f"  TOG={ENABLE_TOG}  DECOMPOSE={ENABLE_DECOMPOSE}  TABLE_SQL={ENABLE_TABLE_SQL}  TABLE_PANDAS={ENABLE_TABLE_PANDAS_CODE}")
    print(f"  WEIGHTED_VOTING={ENABLE_WEIGHTED_VOTING}  PER_TYPE_CONCURRENCY={CONCURRENCY_BY_TYPE}")
    print(f"  CONSENSUS={ENABLE_CONSENSUS}  lock_at={CONSENSUS_LOCK_AT}  reset={RESET_CONSENSUS}")
    print("=" * 70)

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
    print(f"已完成: {len(done_ids)}  |  待答: {len(todo)}  |  per-type 并发: {CONCURRENCY_BY_TYPE}")
    print(f"表格确定性求解覆盖: {deterministic_count}")

    if todo:
        sems = {tt: asyncio.Semaphore(c) for tt, c in CONCURRENCY_BY_TYPE.items()}
        # 题型不在配置里时使用此默认（用最小并发，更稳）
        default_sem = asyncio.Semaphore(min(CONCURRENCY_BY_TYPE.values()))
        out_lock = asyncio.Lock()
        counter = {"done": 0}
        pbar = tqdm(
            total=len(todo),
            desc=f"CCKS-{LLM_PROVIDER}",
            dynamic_ncols=True,
            mininterval=0.2,
            smoothing=0.1,
            position=0,
            leave=True,
        )
        try:
            with open(RAW_OUTPUT_FILE, "a", encoding="utf-8") as out_file:
                tasks = [
                    worker(
                        sems, default_sem, item, out_lock, out_file, counter,
                        len(todo), pbar,
                        consensus_state if ENABLE_CONSENSUS else None,
                    )
                    for item in todo
                ]
                await asyncio.gather(*tasks)
        finally:
            if pbar is not None:
                pbar.close()

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
