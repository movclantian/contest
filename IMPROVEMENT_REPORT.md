# CCKS 2026 OneEval — Pipeline 升级设计报告

> 目标：把当前基线（线上分 ~0.45）提升到更具竞争力的水平。
> 评分由 EM、F1、ROUGE-L 三者均值构成。

---

## 1. 当前基线 (`test.py` PR #1) 状态盘点

| 类别 | 当前做法 | 主要瓶颈 |
| ---- | ----  | ----  |
| knowledge_graph (100 题) | 把全部 ~100 条三元组（最多 230 条）一次性丢给 LLM；3 路自一致投票；带防御式 prompt | **噪声高、干扰项多**（KG-MuLQA Sec 4.2 "Set Operation Failures"、SKA-Bench "Noise Robustness"）。LLM 在长干扰链中容易选错相似实体；多跳推理时遗漏 "NOT/BUT NOT" 约束 |
| multi_hop_qa (100 题) | 把全部 6–8 段上下文丢给 LLM；同样 3 路投票 | **顺序敏感性**与**Lost-in-the-middle**（SKA-Bench "Order Insensitivity"、KG-MuLQA Sec 3）。多跳桥接实体未显式 trace，模型用参数化常识替代 |
| table_qa (50 题) | 启发式确定性 solver 覆盖部分模式 + LLM 回退；表格用 markdown 渲染 | **算术/聚合**类问题 LLM 易错（TabReX 强调结构感知）；solver 模式覆盖不全，剩余靠 LLM 直推 |
| 全局 | 固定 thinking_budget=10000；temperature=0.3；单次问答+一次 repair | **不分难度统一处理**（OneEval 结论 c："diminishing, eventually negative, returns from extended reasoning chains"）；缺少**evidence-based verification** |

---

## 2. 论文要点 → 算法策略 映射

| 论文 | 关键结论 | 我们落地的策略 |
| ---- | ----  | ----  |
| **OneEval** (2506.12577) | 结构化复杂度↑ → 准确率断崖式下跌；推理链过长边际递减且最终负收益 | 按题型自适应 thinking_budget；KG/Table 用更短链 + 更强检索 |
| **KG-MuLQA** | 错误集中在 Set Operations、Implicit Information Gaps、Semantic Misinterpretation；Oracle vs Full Document 对 hard 题反转 | KG 端做**子图检索**而非"oracle 单 triple"；prompt 中强制声明 AND/OR/NOT 集合运算 |
| **KGQAGen-10k** | 迭代式 LLM-Guided Subgraph Expansion 是高质量 QA 构造路径 | 用同样思想做**问答时的 BFS 子图扩展**：从问题实体出发，按相关谓词加权扩展 2 跳 |
| **SKA-Bench** | 四个核心能力：Noise Robustness、Order Insensitivity、Information Integration、Negative Rejection；DeepSeek-R1 在 noise/order 上敏感 | (a) 子图/段落预检索降噪 (b) **顺序扰动**式自一致投票 (c) 显式禁止 "I don't know" 类回答 (d) **citation verification** 防止幻觉 |
| **TabReX** | Text-to-Graph 表格抽取 + 结构对齐评分；强调单元格级正确性 | 表格用严格 Markdown 渲染 + 扩展确定性 solver + 把表格语义化为"row-as-record"列表喂给 LLM |
| **Chain-of-Table** (ICLR'24, 补充) | 在推理链中迭代演化表格（select / group_by / filter）作为中间状态 | table_qa fallback 走"列出候选行 → 算术"两阶段 prompt |
| **StepChain GraphRAG** (Oct 2025, 补充) | 子问题分解 + BFS 检索 → MuSiQue / HotpotQA SOTA | 对多跳/复杂 KG 题做**子问题分解**两段式：(1) decompose+retrieve (2) answer |
| **Socratic Questioning** (EMNLP'23) | 递归子问题分解显著提升 MMLU/MATH/LogiQA | 难题（带"both/and/not/which/exclusive"）走递归式 plan-then-solve |

---

## 3. 新 Pipeline 总览

```
┌────────────────────────────────────────────────────────────────────────┐
│ 1. PREPROCESS (按 task_type)                                          │
│    KG          → parse_triples → entity_index → fuzzy_match_question  │
│                 → BFS_subgraph(seed, depth=2, cap=60) + 抽样剩余 noise │
│    multi_hop   → BM25 score against question/entities                 │
│                 → 保留 top-K 段落（K=6/all）                          │
│    table       → 扩展 deterministic solver；命中则跳过 LLM           │
├────────────────────────────────────────────────────────────────────────┤
│ 2. SELF-CONSISTENCY VOTING (N=5/3, 多样性扰动)                        │
│    - 4-5 路并行调用，温度 [0.20, 0.30, 0.40, 0.50, 0.25]               │
│    - 每路对上下文做"顺序扰动"（shuffle 三元组/段落）→ SKA-Bench       │
│    - 一路用 "be direct"，一路用 "step-by-step"，一路用 "list evidence"  │
│    - 每路独立解析 → 候选池                                            │
├────────────────────────────────────────────────────────────────────────┤
│ 3. SEMANTIC MAJORITY VOTE                                              │
│    - 按 answer 类型 (count / year / entity / list / yes_no) 归一化     │
│    - 数字题：解析数字后众数                                            │
│    - 实体题：rapidfuzz 阈值 ≥85 视为同一实体；众数                     │
├────────────────────────────────────────────────────────────────────────┤
│ 4. EVIDENCE-BASED VERIFICATION (额外 1 路)                            │
│    - 把 top 候选 + 原 context 发给模型："This proposed answer is X.   │
│      Cite the EXACT triple/sentence from the source that supports it. │
│      If unsupported, output the correct answer with citation."         │
│    - 若 verifier 给出 "REVISE: <new>" 且新答案在源文档中可定位，采用  │
├────────────────────────────────────────────────────────────────────────┤
│ 5. TYPE-AWARE POSTPROCESS                                              │
│    - detect_answer_type(question) → coerce 答案                       │
│    - "how many" → 抽取整数；"what year" → 4-digit year                │
│    - 去除 "Answer: " "The answer is" 等前缀                           │
│    - 多答案/列表题：按源文件中相同顺序串接                            │
├────────────────────────────────────────────────────────────────────────┤
│ 6. REPAIR & RECOVERY (兜底)                                            │
│    - 若仍为空/拒答 → 升温 0.7 + 强制不拒答 prompt 重答                │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 4. 关键实现细节

### 4.1 KG 子图检索 (BFS Subgraph Expansion)

```text
def extract_subgraph(triples, question, depth=2, max_size=60):
    1. 提取所有实体（head + tail）；建立 entity→[triples] 倒排索引
    2. 用 rapidfuzz 在 question 中匹配 top-N 命名实体 (>=85 partial_ratio)
    3. 从种子实体出发 BFS depth 跳，按关系/实体频次去重
    4. 截断 max_size；若空间够，再补一些"关系命中"的三元组
    5. 顺序：按 (depth, relation_relevance) 排序
```

### 4.2 多跳段落 BM25 检索

```text
def rank_passages(question, contexts, top_k=6):
    1. 对每段落计算 BM25(question, paragraph)
    2. 额外加分：包含问题中专有名词的段落 +α
    3. 取 top_k；若 K >= n，全部保留但重排序
```

### 4.3 顺序扰动自一致投票

```text
async def vote(item, n=5):
    seeds = [0, 1, 2, 3, 4]
    temps = [0.20, 0.30, 0.40, 0.50, 0.25]
    styles = ["direct", "cot", "evidence", "decompose", "verify-self"]
    
    每路：
      - random.seed(seed) 对 triples/contexts shuffle
      - 用对应 prompt style + temperature
      - 收集答案 + 简短理由
    
    semantic_majority(answers)
```

### 4.4 类型感知后处理

| 问题模式 | 答案归一化 |
| ---- | ----  |
| "how many / count / number of" | 抽取整数 |
| "in what year" | 4-digit 年份 |
| "on what date" | 完整日期或月日 |
| "which (entity/film/person/team)" | 去引号、去 "the"、保留完整专有名词 |
| "yes/no / does / is / was" | "Yes" / "No" |
| 列表题（"list all / which ... and ..."）| 按源中顺序串接，逗号分隔 |

### 4.5 扩展 Deterministic Table Solver

补充以下模式（基于 50 道 table_qa 题面分析）：
- "average / mean of X" → 数值列 mean
- "median X" / "median value"
- "highest / lowest / top X" → max/min
- "ratio of A to B" → 两个分组聚合的比值
- "before / after Y" → 行时序过滤
- "rank N" → 按数值排序取第 N

---

## 5. 预期收益（按论文对照）

| 改动 | 直接受益的论文结论 | 预期影响 |
| ---- | ----  | ----  |
| KG 子图检索 | KG-MuLQA Set Op、SKA-Bench Noise Robust | KG +5~10 分（噪声 80% → 20%） |
| 段落 BM25 | SKA-Bench Order Insensitivity | 多跳 +3~6 分 |
| 顺序扰动 self-consistency 5 路 | SKA-Bench Order + Wang-2022 Self-Consistency | 所有题 +2~4 分 |
| Evidence-based verification | KG-MuLQA Implicit Gaps、SKA-Bench Negative Rejection | 减少幻觉，全局 +2~3 分 |
| 自适应 thinking budget | OneEval 推理链边际递减 | 减少过度发散错误，全局 +1~2 分 |
| Type-aware 后处理 | F1/ROUGE-L 直接相关 | EM/F1 各 +1~3 分 |
| Table solver 扩展 | TabReX 单元格对齐 | table_qa +5~10 分 |

合并估计：**0.45 → 0.55+ 仍可期**，但实际增益依赖 deepseek-v4-pro 在长上下文上的表现。

---

## 6. 风险与回退

- **API rate limit**：投票从 3 → 5，调用量↑ 67%。已保留并发=80 与指数退避，OK。
- **Verification 误改**：verifier 只在它能给出"具体证据片段"时才采纳；否则保留多数投票结果。
- **子图召回失败**：当 fuzzy match 找不到种子实体时回退到全量三元组+排序。

---

## 7. Provider 切换（Anthropic / OpenAI 二选一）

通过环境变量 `LLM_PROVIDER` 单一开关切换两套 SDK，两边都按各自的"拉满"模式配置：

| 维度 | `LLM_PROVIDER=anthropic` (默认) | `LLM_PROVIDER=openai` |
| ---- | ----  | ----  |
| SDK | `anthropic.AsyncAnthropic` | `openai.AsyncOpenAI` |
| 推荐端点 | `https://api.deepseek.com/anthropic` | `https://api.deepseek.com` (DeepSeek OpenAI 兼容) 或 `https://api.openai.com/v1` |
| 推理预算 | `thinking={"type":"enabled","budget_tokens":N}` (按题型 8k-12k) | `reasoning_effort="max"` (DeepSeek) / `reasoning.effort="high"` (gpt-5) |
| 力度控制 | `output_config={"effort":"max"}` | `reasoning_effort="max"` + `text.verbosity="high"`（仅 Responses API） |
| temperature | 0.3 (thinking 模式下被服务端忽略，但不报错) | 同左；通过 prompt 风格/顺序扰动来获取多样性 |
| 调用方法 | `client.messages.create(...)` | `client.chat.completions.create(...)` 或 `client.responses.create(...)` |
| 选择 Responses API | 不适用 | `OPENAI_USE_RESPONSES_API=true`（gpt-5/o3/o1 系列） |
| DeepSeek thinking 透传 | 默认开启（Anthropic 原生） | `extra_body={"thinking":{"type":"enabled"}}`（默认开启） |

切换方式（单次调用）：
```bash
LLM_PROVIDER=anthropic python3 test.py   # 默认
LLM_PROVIDER=openai    python3 test.py   # 改走 OpenAI SDK
```

两个 provider 完全共享上层的：KG 子图、BM25、多样性投票、verification、repair、确定性 table solver、类型感知后处理。

---

## 8. 后续可选 (Out-of-scope 本 PR)

- **Pandas 沙盒**：让 LLM 输出 pandas 表达式 → 安全 ast 评估（高 ROI 但代码复杂度大）。
- **Few-shot exemplar**：从 250 题中挑 3 题作为 in-context demo（需要观察实际错例）。
- **Cross-task agreement**：多模型 ensemble（GLM4 + DeepSeek + Qwen3）。
