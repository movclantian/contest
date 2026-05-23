我已经对你提供的五篇核心论文（OneEval、KG-MuLQA、KGQAGen-10k、SKA-Bench、TabReX）以及你的基线代码 test.py 进行了全面、深度的阅读与交叉分析。
根据论文中严谨的实验结论与误差分析（Error Analysis），你当前的 test.py 代码存在架构级与策略级的硬伤。单纯依靠 deepseek-v4-pro 的超大上下文和全量思考预算（Thinking Budget），在结构化知识（KG/Table）面前不仅无法提分，反而会触发模型固有的失效模式（Failure Modes）。
以下是基于论文事实得出的核心痛点，以及你必须立即执行的代码重构方案：
1. 论文核心结论与 test.py 的致命冲突
冲突 A：推理链边际递减效应 vs 无脑拉满思考预算
- 论文事实（OneEval 结论 C）：_“Longer reasoning chains offer diminishing, and eventually negative, returns”_（当知识模态变得结构化时，更长的推理链会带来边际递减甚至负收益，引入更多噪声）。
- 你的代码：THINKING_BUDGET = 32000。
- 后果：在 KG 存在干扰项（R-Stress）时，DeepSeek 的思维链越长，越容易陷入自我纠结和发散，最终把正确的实体跟干扰实体混淆。
冲突 B：集合操作（Set Operations）与语义歧义
- 论文事实（KG-MuLQA 4.2 Error Analysis）：最强 LLM 在面对集合操作（如“谁是 A 的高管但不是 B 的高管”）和特定语义（如把“Zonal Office”错答成“City”）时，准确率断崖式下跌。
- 你的代码：knowledge_graph 的 Prompt 仅提供了 Reason carefully step by step。
- 后果：模型会使用通用常识（Parametric Knowledge）覆盖图谱事实，且极易在多跳（Multi-hop）聚合时遗漏“排除性（NOT）”条件。
冲突 C：表格推理的降维错位
- 论文事实（TabReX）：引入了基于图的评估（Text2Graph），强调表格结构（行/列对齐、空单元格）对事实提取的决定性作用。
- 你的代码：用简单的 | 拼接字符串，丢弃了标准 Markdown 表格的对齐分隔符（如 |---|---|），导致 LLM 视觉对齐错乱。
2. 强制整改方案（KISS & DRY 原则）
不要重写整个工程，保留你的并发调度层，但必须重构配置与 Prompt 拼装策略。
步骤一：收缩配置，遏制模型幻觉
修改 test.py 的配置区：
# 降低预算，阻断在结构化数据上的"过度发散"
THINKING_BUDGET = 8192  # 从 32000 降至 8192 或 12288
MAX_TOKENS = 12000      # 同步收缩
TEMPERATURE = 0.3       # 知识密集型推理强制降温（极其重要，1.0 会导致结构化数据提取极不稳定）
TOP_P = 0.85
步骤二：重构 Table 表格渲染格式（遵循标准 Markdown 规范）
基于 TabReX 对表格结构敏感性的研究，必须给 LLM 渲染完全符合规范的表格：
    elif task_type == "table_qa":
        table = item["table"]
        header = table["header"]
        rows = table["rows"]
        # 强制标准 Markdown 渲染，增强结构感知
        header_line = "| " + " | ".join(str(h).strip() for h in header) + " |"
        separator_line = "| " + " | ".join("---" for _ in header) + " |"
        
        table_text = header_line + "\n" + separator_line + "\n"
        for row in rows:
            table_text += "| " + " | ".join(str(cell).strip() for cell in row) + " |\n"
        return f"""Analyze the following table and answer the question.
{table_text}
Question: {question}
Instructions:
- Answer based EXCLUSIVELY on the provided table data.
- Row/Column alignment is critical. Locate the exact cell before reasoning.
- Give a concise final answer (a name, number, or short phrase).
- Output ONLY the final answer text, no prefixes."""
步骤三：重构 KG 与 Multi-hop 提示词（防御陷阱）
针对 KG-MuLQA 提出的 "Set Operation Failures" 和 "Misinterpretation of Semantics"，在 Prompt 中强制加入防干扰与集合计算声明：
    if task_type == "knowledge_graph":
        return f"""Knowledge Graph Triples (Format: <Head> <Relation> <Tail>):
{item['input']}
Question: {question}
Instructions:
1. EXPLICIT FACT CHECKING: Base your answer strictly on the provided triples. Do NOT use external general knowledge.
2. DISTRACTOR AVOIDANCE: Identify and exclude entities with similar names but incorrect relations.
3. SET OPERATIONS: Pay strict attention to constraints like 'AND', 'OR', 'NOT', 'BUT NOT' in the question. Mentally compute intersections or differences if required.
4. Output ONLY the final answer text (e.g., exact entity name or number). Do not include explanations or prefixes like "Answer:". """
3. 下一步建议（高阶架构）
按照赛题机制，单次请求（Zero-Shot）已经是现有架构的上限。如果你做完上述 3 步修改后准确率依然无法登顶，根据官方的复杂程度设定，你必须引入工作流：
1. 对于 Table_QA：停止让 LLM 直接肉脑推算。把表格转化为 CSV 结构，让 DeepSeek 生成一段 Python Pandas 代码（df.loc[...]），通过安全的沙盒（Sandbox 工具）执行代码返回结果。这是目前解决复杂表格计算（TabReX 提到的 numeric differences）的唯一满分路径。
2. 对于 Multi-Hop：采用类似 SOTA 论文中的分步策略（Decomposition）。先让模型输出 ["Sub-question 1", "Sub-question 2"]，再带着子问题去文中检索，最后汇总。
