# **OneEval: Benchmarking LLM Knowledge-intensive Reasoning over Diverse Knowledge Bases** 

**Yongrui Chen[1,2] , Zhiqiang Liu[3] , Jing Yu[3] , Lin Ren[1,2] , Nan Hu[1,2] , Xinbang Dai[1,2] , Jiajun Liu[1,2] , Jiazhen Kang[1,2] , Shenyu Zhang[1,2] , Xinda Wang[3] , Keyan Ding[3] , Pengfei Shen[4] , Haolei Zhu[4] , Hongjie Deng[3] , Yisong Wang[5] , Tongtong Wu[6] , Sheng Bi[1] , Wen Zhang[3] , Tianxing Wu[1,2] , Qiu Ji[4] , Haofen Wang[5] , Wenliang Chen[7] , Huajun Chen[3] , Guilin Qi[1, 2]** 

1Southeast University, China 

2Key Laboratory of New Generation Artificial Intelligence Technology and its Interdisciplinary Applications (Southeast University), Ministry of Education, China 

3Zhejiang University, China 

4Nanjing University of Posts and Telecommunications, China 

5Tongji University, China 

6Monash University, Australia 

7Soochow University, China `{yongruichen, gqi}@seu.edu.cn` 

## **Abstract** 

Large Language Models (LLMs) have demonstrated substantial progress on reasoning tasks involving unstructured text, yet their capabilities significantly deteriorate when reasoning requires integrating structured external knowledge such as knowledge graphs, code snippets, or formal logic. This limitation is partly due to the absence of benchmarks capable of systematically evaluating LLM performance across diverse structured knowledge modalities. To address this gap, we introduce **ONEEVAL** , a comprehensive benchmark explicitly designed to assess the knowledge-intensive reasoning capabilities of LLMs across four structured knowledge modalities—unstructured text, knowledge graphs, code, and formal logic—and five critical domains (general knowledge, government, science, law, and programming). ONEEVAL comprises 4,019 carefully curated instances and includes a challenging subset, ONEEVALHard, consisting of 1,285 particularly difficult cases. Through extensive evaluation of 18 state-of-the-art open-source and proprietary LLMs, we establish three core findings: a) _persistent limitations in structured reasoning_ , with even the strongest model achieving only 32.2% accuracy on ONEEVALHard; b) _performance consistently declines as the structural complexity of the knowledge base increases_ , with accuracy dropping sharply from 53% (textual reasoning) to 25% (formal logic); and c) _diminishing returns from extended reasoning chains_ , highlighting the critical need for models to adapt reasoning depth appropriately to task complexity. We release the ONEEVAL datasets, evaluation scripts, and baseline results publicly, accompanied by a leaderboard to facilitate ongoing advancements in structured knowledge reasoning. 

## **1 Introduction** 

Large Language Models (LLMs) [1, 2, 3] have recently demonstrated significant progress in complex reasoning tasks. Capabilities like synthesizing reasoning paths from unstructured text and leveraging implicit commonsense knowledge have advanced considerably [4, 5], as evidenced by performance on various benchmarks [6, 7, 8, 9]. However, even for models explicitly optimized for reasoning, 

Preprint. Under review. 

**==> picture [396 x 195] intentionally omitted <==**

**----- Start of picture text -----**<br>
5 Domains 4 Knowledge Bases<br>OneEval<br>General Law Text Logic<br>OneEval-Hard<br>Government Science Program Code Knowledge Graph AnalysisHuman<br>Task Performance<br>Input Format Output Format Metric<br>11 ReasoningTasks Question knowledge Answer F1-score KB Performance<br>Statement Triple Accuracy Domain Performace<br>Code Code<br>LLM ISM@1 Case Study<br>Empirical Conclusions<br>… 18 Advanced LLMs<br>… …<br>**----- End of picture text -----**<br>


Figure 1: Overall framework of ONEEVAL, covering 4 knowledge bases, 5 fields, and 11 tasks. 

LLMs continue to exhibit significant fragility and inaccuracy when required to integrate structured external knowledge bases such as code, knowledge graphs, or formal logic [10]. 

A critical limitation underlying this challenge is the nature of existing reasoning benchmarks. They predominantly focus on unstructured textual reasoning [11, 12], which primarily involves synthesizing information from narrative text. In contrast, many practical reasoning tasks necessitate the processing and integration of structured knowledge formats such as codes, KGs, or formal logical statements. These formats differ fundamentally from the unstructured text that constitutes the majority of LLM training data. Consequently, the performance landscape of state-of-the-art LLMs—including powerful models like DeepSeek R1 [13], Grok3 [14], and o3 [15]—across diverse knowledge modalities (e.g., formal logic, symbolic manipulation), and how performance trends change when transitioning from familiar textual distributions to these structured forms [10], remains inadequately characterized. 

Motivated by this critical need for a comprehensive evaluation framework, we introduce **ONEEVAL** , the first unified benchmark specifically designed to assess LLM reasoning capabilities across a spectrum of knowledge modalities. As illustrated in Figure 1ONEEVAL covers unstructured text, knowledge graphs, code, and formal logic, spanning five high-impact domains (general knowledge, government, science, law, and programming). Comprising 4,019 instances meticulously curated from 11 meticulously curated dataset, the benchmark also includes a ONEEVALHard subset of 1,285 particularly challenging cases. Each instance provides a query paired with an external knowledge base presented in one of the specified modalities, alongside a gold answer. 

A large-scale evaluation campaign involving 18 diverse open-source and proprietary models (Table 2) yields three robust findings concerning the current state of LLM reasoning on structured knowledge: a) _Current LLMs remain significantly challenged by knowledge-intensive reasoning_ : Even the most capable model achieves merely 32.2% accuracy on the challenging ONEEVALHard subset, indicating substantial room for improvement in reliability. b) _Reasoning performance degrades consistently as the knowledge modality becomes more structured_ : Average accuracy declines sharply from 53% on unstructured text to 25% on formal logic, revealing a clear dependency on the input structure. c) _Longer reasoning chains offer diminishing, and eventually negative, returns_ : Beyond a moderate length, the introduction of noise and potential errors outweighs the benefits of additional steps, highlighting the need for chain-length sensitivity in complex reasoning across modalities. Our key contributions include: 

- **Benchmark.** We release ONEEVAL, the first comprehensive LLM knowledge-intensive reasoning benchmark spanning four types of knowledge bases, five domains, 4,019 instances, and a 1,285sample hard subset, together with unified evaluation scripts. 

2 

- **Analysis.** We present a comprehensive and detailed study encompassing 18 advanced LLMs. Our work reveals crucial insights into the interplay between factors such as the degree of knowledge structuring, model size, and response length, and their impact on LLM reasoning performance. 

- **Resources.** All datasets, prompts, and backbone LLM outputs are publicly available, and an online leaderboard encourages rapid community progress on knowledge-intensive reasoning. 

## **2 ONEEVAL** 

We introduce ONEEVAL, a comprehensive benchmark designed to rigorously evaluate the knowledge reasoning capabilities of LLMs across a spectrum of knowledge base (KB) types and reasoning challenges. ONEEVAL is constructed from 15 distinct public datasets, carefully selected to represent diverse facets of knowledge, including structured, unstructured, explicit, and implicit forms. To create a manageable yet representative evaluation set, for each constituent dataset _D[i]_ (where _i ∈ {_ 1 _, . . . ,_ 15 _}_ ), we randomly sample _N[i]_ instances. The final ONEEVAL benchmark comprises the union of these sampled instances. Detailed statistics of the selected datasets, including their sources and sampling sizes _N[i]_ , are provided in Table 1. Further specifics regarding the source datasets and their original task formats can be found in the Appendix. 

## **2.1 Task Definition** 

Given a user query _Q_ and an accessible knowledge base _S_ , the objective is to generate the desired answer _A_ by leveraging information available within _S_ . Formally, the task for an LLM _fθ_ is to compute _A_ = _fθ_ ( _Q, S_ ). Here, the query _Q_ can be presented in various formats, including natural language questions, statements, descriptions, or code snippets. The knowledge base _S_ is drawn from a predefined set of distinct types, which we detail in the following subsection. The answer _A_ should be a valid derivation or inference based on the provided _Q_ and _S_ , and its format can vary, encompassing free-form text, structured outputs like triples, boolean values, or code snippets. 

## **2.2 Knowledge Base Types** 

A critical aspect distinguishing ONEEVAL is its focus on evaluating LLM performance conditioned on diverse types of external knowledge bases. This allows for a nuanced analysis of their capabilities across different data structures and reasoning paradigms. We categorize the knowledge bases in ONEEVAL into five principal types, forming the set ΩKB = _{D, T , K, C, L}_ , where: 

- **Textual Base (** _D_ **):** A textual base _D_ = _{d_ 1 _, d_ 2 _, . . . , dn}_ represents an unstructured collection of natural language documents or passages, where each _di_ is a unit of text. 

- **Knowledge Graph (** _K_ **):** A knowledge graph (KG) _K_ is a structured semantic network typically defined as a set of triples _K_ = _{⟨s, p, o⟩|s ∈E, p ∈R, o ∈E ∪L}_ , where _E_ is the set of entities, _R_ is the set of relations, and _L_ is the set of literal values. 

- **Code Bases (** _C_ **):** A code base _C_ = _{c_ 1 _, c_ 2 _, . . . , cm}_ represents a collection of programmatic knowledge artifacts. Each _ci_ is an element such as source code snippets, function definitions, API documentation entries, or specifications. 

- **Logic Bases (** _L_ **):** A logic base _L_ provides a formal, explicit specification of a domain’s conceptualization. It is typically defined by a tuple _L_ = ( _Cont, Pont, Aont_ ), where _Cont_ is the set of concepts (classes), _Pont_ is the set of properties (relations between concepts), and _Aont_ is a set of axioms or rules defining constraints and logical relationships. 

## **2.3 Domain** 

ONEEVAL spans five key knowledge domains, General, Government Affairs, Science, Law, and Programming, with a strong emphasis on the breadth and specialization of multi-source heterogeneous knowledge. It aims to systematically evaluate LLMs in terms of reasoning and application capabilities in complex, knowledge-driven tasks. The specific domain categories are illustrated in Appendix. 

3 

Table 1: Statistics of each dataset in ONEEVAL. 

|**Knowledge Base**<br>Text<br>Knowledge<br>Graph<br>Code<br>Logic|**Task**<br>BioTextQA<br>MatTextQA<br>ChineseLawFact<br>AttributionNLI<br>KCQAD<br>PharmKGQA<br>AffairQA<br>PeopleRelQA<br>ReportFixer<br>VersiCode<br>SymTex-ASC|**Input**<br>**Output**<br>**Language**<br>**# of Samples**<br>**Metric**|
|---|---|---|
|||Question<br>Answer<br>English<br>210<br>F1<br>Question<br>Answer<br>English<br>210<br>F1<br>Description<br>Boolean<br>Chinese<br>800<br>Acc<br>Question<br>Answer<br>English<br>210<br>F1<br>Question<br>Answer<br>English<br>500<br>F1|
|||Question<br>Answer<br>English<br>210<br>F1<br>Question<br>Answer<br>Chinese<br>200<br>Acc<br>Question<br>Answer<br>Chinese<br>200<br>F1<br>Description<br>Triple<br>Chinese<br>200<br>F1|
|||Code<br>Code<br>English<br>739<br>ISM@1|
|||Statement<br>Code<br>English<br>540<br>EM|



## **2.4 ONEEVAL-Hard** 

To enable a more accurate and fine-grained evaluation of LLM performance, particularly in highdifficulty reasoning scenarios, we have manually curated a challenging subset of ONEEVAL, denoted as ONEEVAL-Hard. The construction of ONEEVAL-Hard involved a multi-stage process combining empirical performance analysis with expert qualitative review. 

For each test sample ( _Q, S, A_ ) from the full ONEEVAL, we first empirically assessed its difficulty by evaluating a set of _K_ LLMs, _F_ = _{fθ_[1] _[, . . . , f] θ[ K][}]_[.][Let] _[ σ]_[(] _[A][, f][θ]_[(] _[Q][,][ S]_[))] _[∈{]_[0] _[,]_[ 1] _[}]_[ be an indicator] variable, where _σ_ = 1 if _fθ_ correctly answers sample _x_ , and _σ_ = 0 otherwise. We define the empirical hardness score of sample ( _Q, S, A_ ) as the proportion of _fθ_ in _F_ that fail to answer it correctly: 

**==> picture [287 x 30] intentionally omitted <==**

Samples with a high empirical hardness score _H_ ( _x_ ) were identified as candidates for ONEEVAL-Hard. 

Subsequently, these candidates underwent multiple rounds of expert screening and review. Human experts qualitatively analyzed the samples, prioritizing those that specifically demand sophisticated reasoning capabilities known to be challenging for current LLMs. This includes instances requiring multi-step logical deduction, the association of implicitly stated knowledge, and the synthesis of information from disparate sources (cross-domain knowledge integration). 

The final ONEEVAL-Hard subset consists of 1,285 samples selected through this rigorous filtering process. By focusing on samples that empirically challenge existing models and qualitatively require complex reasoning patterns, ONEEVAL-Hard offers higher discriminative power and represents a more challenging testbed. It serves as a critical resource for pinpointing specific knowledge blind spots and reasoning bottlenecks in LLMs, thereby providing an important foundation for driving targeted model analysis, optimization, and capability enhancement. 

## **3 Experimental Setup** 

During the evaluation phase of ONEEVAL, the parameters _θ_ of the target LLM _fθ_ are kept constant. Each test sample’s prompt is generated by integrating the user input _Q_ with the retrieved external knowledge set _S_ , which is converted into a textual format to be compatible with the input requirements of _fθ_ . The full prompt construction details of each task are provided in the Appendix. 

**External Knowledge Retrieval Paradigm.** As retrieval capability is not the focus of ONEEVAL, the external knowledge set _S_ for each test sample is obtained through a standardized retrieval approach and remains consistent across different instances of _fθ_ . Specifically, leveraging dense retrieval techniques, the core methodology involves ranking knowledge fragments—such as text paragraphs, code snippets, or subgraphs of triples—based on the similarity _σ_ ( _Q, S_ ) between the dense vector representation of _Q_ and _S_ , _σ_ ( _Q, S_ ) = cos( **q** _,_ **s** ). Notably, the retrieved knowledge context may contain noise, reflecting real-world scenarios and providing a robust testbed to evaluate the model’s resilience to imperfect or redundant information. 

4 

Table 2: Overall Score (%) of ONEEVAL and ONEEVAL-Hard. 

|**Models**|**Release Date**|**Size**|**R-LLM**|**ONEEVAL**|**ONEEVAL-Hard**|∆**Hard**|
|---|---|---|---|---|---|---|
|_Open-Source LLMs_|||||||
|Qwen2.5-7B|Sep 2024|7B||33_._3|10_._7|_−_22_._6|
|Llama3.1-8B|Jul 2024|8B||31_._0|10_._3|_−_20_._7|
|GLM4-9B|Jun 2024|9B||39_._7|13_._2|_−_26_._5|
|QWQ-32B|Mar 2025|32B|✓|44_._9|15_._5|_−_29_._4|
|Llama3.1-70B|Jul 2024|70B||45_._8|15_._9|_−_29_._9|
|Qwen2.5-72B|Sep 2024|72B||46_._0|16_._1|_−_29_._9|
|Llama4-Maverick|Apr 2025|400B||**48.2**|**21.0**|_−_27_._2|
|DeepSeek-V3|Dec 2024|671B||44_._6|17_._8|_−_26_._8|
|DeepSeek-R1|Jan 2025|671B|✓|47_._2|17_._2|_−_30_._0|
|_Proprietary LLMs_|||||||
|GPT-4o|May 2024|-||46_._1|16_._3|_−_29_._8|
|GPT-4.1|Apr 2025|-||58_._1|24_._7|_−_33_._4|
|o1|Dec 2024|-|✓|-|22_._2|-|
|o3|Apr 2025|-|✓|-|**32.2**|-|
|o4-mini|Apr 2025|-|✓|**53.1**|29_._4|_−_23_._7|
|Doubao-pro|Jan 2025|-||42_._1|13_._0|_−_29_._1|
|Claude3.7-Sonnet|May 2025|-|✓|40_._9|15_._2|_−_25_._7|
|Grok3|Feb 2025|-|✓|51_._0|21_._7|_−_29_._3|
|Gemini-2.5-Pro|Mar 2025|-|✓|-|22_._8|-|



**Large Language Models.** We selected a diverse set of representative LLMs, encompassing both opensource and closed-source models, a wide range of parameter scales, and various technical approaches. The models include Llama3.1-8B [16], Llama3.1-70B [16], GLM4-9B [17], Qwen2.5-7B [18], Qwen2.5-72B [18], QWQ-32B [19], DeepSeek-V3 [3], DeepSeek-R1 [13], Llama4-Maverick [20], GPT-4o[1], GPT-4.1 [21], o1 [2], o3 [15], o4-mini [15], Doubao-pro [22], Claude3.7-Sonnet [23], Grok3 [14], and Gemini-2.5-Pro [24]. 

**Evaluation Metric.** In ONEEVAL, we employ multiple evaluation metrics tailored to different tasks, including Accuracy, F1 score, and ISM@1. Detailed metrics for each task are outlined in Table 2. To ensure a fair assessment of a model’s overall performance across tasks, we follow established LLM benchmarks and define the Overall Score as the average of its scores across all evaluation datasets. 

## **4 Results & Analysis** 

## **4.1 Overall results** 

Table 2 presents the overall accuracy of a diverse set of open-source and proprietary LLMs on both the standard ONEEVAL and ONEEVAL-Hard benchmarks. Due to space limitations, detailed experimental results for each dataset can be found in the Appendix. Among the open source models, Llama4-Maverick achieved the best results thanks to its hybrid expert architecture. Among the private models, o4-mini and o3 achieved the best results on full set and hard set, respectively. In general, as the model size increases, the inference effect gradually improves. All models experience a substantial performance degradation when faced with the more challenging subset. For instance, the highest overall score on ONEEVAL-Hard among open-source models (Llama4-Maverick, 21.0%) remains markedly lower than its standard ONEEVAL performance (48.2%), corresponding to a drop of 27.2 percentage points. Similarly, proprietary models such as GPT-4.1 and o3, although leading on ONEEVAL-Hard (24.7% and 32.2% respectively), still demonstrate large absolute gaps compared to their performance on the easier benchmark. The pronounced and universal decline across all models—often exceeding 25 percentage points—underscores the significant unresolved challenges. Therefore, we conclude **Insight 1: Even current state-of-the-art reasoning LLMs exhibit significant limitations when tackling knowledge-intensive reasoning tasks.** 

5 

**==> picture [396 x 318] intentionally omitted <==**

**----- Start of picture text -----**<br>
Full Set Performance<br>70 Textual Reasoning<br>60 Code ReasoninKG Reasoning g<br>50 Logic Reasoning<br>40<br>30<br>20<br>10<br>0<br>Hard Set Performance<br>50<br>40<br>30<br>20<br>10<br>0<br>Qwen2.5-7BLlama3.1-8B GLM4-9B QWQ-32BLlama3.1-70BQwen2.5-72BLlama4-MaverickDeepSeek-V3DeepSeek-R1Doubao-proClaude-3.7-Sonnet GPT-4o Grok-3 GPT-4.1 o4-mini<br>Qwen2.5-7Bllama3.1-8BGLM4-9BQWQ-32Bllama3.1-70BQwen2.5-72BDeepSeek-V3DeepSeek-R1Llama4-MaverickDoubao-proClaude-3.7-SonnetGemini-2.5-Pro GPT-4o Grok-3 GPT-4.1 o1 o4-mini o3<br>Score (%)<br>Score (%)<br>**----- End of picture text -----**<br>


Figure 2: Performance across different knowledge bases for the Full set (top) and Hard set (bottom). 

## **4.2 Performance on different KBs** 

Figure 2 compares model performance across different knowledge bases for both the Full (top) and Hard (bottom) subsets of ONEEVAL. Overall, Textual Reasoning tasks yield the highest scores, particularly for larger models such as GPT-4.0, GPT-4.1, and Q4-mini, suggesting that these tasks are relatively straightforward for current architectures. In contrast, Logic and Knowledge Graph (KG) Reasoning remain challenging, especially on the Hard subset, where all models exhibit pronounced performance drops. Among code tasks, GPT-4.1 achieves the best results on the Full set, while Gemini-2.5-Pro and o3 show strong performance on the hardest instances. DeepSeek-V3 stands out on Logic Reasoning tasks, and Grok3 and o3—both designed for advanced reasoning—lead the KG benchmarks on the Full and Hard sets, respectively. Furthermore, larger models consistently outperform smaller counterparts such as Qwen2.5-7B and Llama-3-8B, particularly in code and textual reasoning. However, even state-of-the-art models struggle with logic and KG tasks, underscoring current limitations in abstract and structured reasoning. 

## **4.3 LLM Performance with Increasing Knowledge Structuredness** 

In our exploration of performance trends and rules governing LLMs in reasoning tasks across different types of KBs, we organized the four KBs in ONEEVAL by increasing levels of structuredness: text, code, knowledge graphs, and logic. We then analyzed the performance trends of various models on both the full and hard sets of these tasks. The experimental results are depicted in Figure 3. Here, AVERAGE, AVERAGE-R, and AVERAGE-NR represent the average of all model performances, the average of R-LLM performances, and the average of non-inferential LLM performances, respectively. 

First, we point out **Insight 2: As the level of structure in tasks increases, the reasoning performance of LLMs tends to decline, indicating challenges in handling highly organized information.** . 

6 

**==> picture [396 x 110] intentionally omitted <==**

**----- Start of picture text -----**<br>
80 Full Set Hard Set Qwen2.5-7B<br>70 50 llama3.1-8BGLM4-9B<br>QWQ-32B<br>60 40 llama3.1-70BQwen2.5-72B<br>Llama4-Maverick<br>50 DeepSeek-V3<br>40 30 DeepSeek-R1Doubao-proClaude-3.7-Sonnet<br>GPT-4o<br>30 20 Grok-3<br>Gemini-2.5-Pro<br>20 GPT-4.1o1<br>10 o4-mini<br>10 o3<br>AVERAGE<br>Textual Reasoning0 Code Reasoning KG Reasoning Logic Reasoning Textual Reasoning0 Code Reasoning KG Reasoning Logic Reasoning AVRAGE-RAVRAGE-NR<br>Score (%) Score (%)<br>**----- End of picture text -----**<br>


Figure 3: Score (%) of ONEEVAL and ONEEVAL-Hard with Increasing Knowledge Structuredness. 

On the full set, the average score drops from 53 % on textual reasoning to 25 % on logic reasoning; on the hard subset the decline is even sharper (from 23 % to 14 %). The knee of the curve is the transition from code to KG reasoning: although most models clear 50 % on code problems, they typically lose 25–30 absolute points when explicit graph navigation is required. This suggests that, while current instruction-tuned models have internalized a fair amount of “latent code semantics,” they still struggle to expose and manipulate discrete graph structures. 

Second, models equipped with explicit inference scaffolds (“R-LLMs”) are systematically more resilient to increasing structure. The red curve (AVERAGE-R) lies above the brown curve (AVERAGENR) by 4.3 points on the full set and by 5.6 points on the hard set, with the gap widening for KG and logic reasoning. In other words, retrieval-augmented or step-by-step inference not only boosts raw accuracy, it scales better as symbolic demands grow. However, the advantage is not unlimited: the R-LLMs still trail human-level performance by a wide margin in logic, implying that retrieval alone is insufficient when deeper deductive chains are necessary. 

Third, variance across models expands with structure. For text and code tasks the best and worst systems differ by about 20 points, whereas in logic reasoning the spread exceeds 30 points in both splits. Outliers are instructive: Grok-3 and Gemini 2.5-Pro maintain relatively high logic scores (32 %), hinting that specialized training on formal proofs or chain-of-thought data can pay off. By contrast, smaller open-weight models like Llama-3-8B collapse to near-random on KG and logic problems, revealing a strong size–capability interaction for symbolic tasks. 

## **4.4 Cross-Modality Transferability** 

**==> picture [227 x 97] intentionally omitted <==**

**----- Start of picture text -----**<br>
Task Correlation Matrix of Full Set  1.0 Task Correlation Matrix of Hard Set 1.0<br>1.00 0.47 0.83 0.83 0.90.8 1.00 0.09 0.51 0.50<br>0.8 0.8<br>0.47 1.00 0.56 0.13 0.70.60.5 0.09 1.00 0.37 0.38<br>0.60.5 0.6<br>0.5<br>0.83 0.56 1.00 0.43 0.4 0.51 0.37 1.00 0.64 0.4<br>0.3<br>0.83 0.13 0.43 1.00 0.50 0.38 0.64 1.00 0.2<br>0.2<br>Text Logic KG Code Text Logic KG Code<br>Text Text<br>Logic Logic<br>KG KG<br>Code Code<br>**----- End of picture text -----**<br>


We compute pairwise Spearmantween model-level accuracies across _ρ_ be1.00 0.47 0.83 0.83 0.90.8 1.00 0.09 0.51 0.50 0.8 four task families and observe a strik-ing split (Figure 4). On the full bench0.47 1.00 0.56 0.13 0.70.60.5 0.09 1.00 0.37 0.38 0.6 mark, Text shows uniformly high 0.83 0.56 1.00 0.43 0.4 0.51 0.37 1.00 0.64 0.4 affinity with both KG ( _ρ_ = 0 _._ 83) and 0.3 Code ( _ρ_ = 0 _._ 83), whereas Logic re0.83 0.13 0.43 1.00 0.2 0.50 0.38 0.64 1.00 0.2 mains largely orthogonal ( _ρ_ = 0 _._ 47 to Text Logic KG Code Text Logic KG Code Text, 0 _._ 13 to Code). The elevated Text _↔_ KG and Text _↔_ Code correlations Figure 4: Correlation coefficients of performance on different tasks. indicate that, for standard-difficulty items, surface-level pattern matching learned from natural language suffices to solve many KG lookup and templated coding questions, yielding a shared representation space. 

Figure 4: Correlation coefficients of performance on different tasks. 

The picture changes on the hard subset, where spurious cues are removed. Text decouples almost entirely from Logic ( _ρ_ = 0 _._ 09) and only modestly aligns with KG/Code ( _≈_ 0 _._ 5), while KG and Code remain mutually supportive ( _ρ_ = 0 _._ 64). This suggests two distinct reasoning substrates: (i) a symbolic substrate spanning KG and Code that survives distribution shift, and (ii) a shallow lexical substrate tying Text to others only when shortcuts are available. Practically, few-shot transfer from KG to Code should be effective even on adversarial data, whereas gains from Text pre-training will collapse under harder regimes—fine-tuning budgets should be allocated accordingly. 

7 

**==> picture [397 x 302] intentionally omitted <==**

**----- Start of picture text -----**<br>
Full Set Overall Results Llama4-Maverick Hard Set Overall Results Llama4-Maverick<br>DeepSeek-R1<br>45 QWQ-32B Llama3.1-70BQwen2.5-72B DeepSeek-V3 2018 DeepSeek-R DeepSeek-V31<br>GLM4-9B 16 QWQ-32B Llama3.1-70BQwen2.5-72B<br>40<br>14 GLM4-9B<br>35Qwen2.5-7B 12Qwen2.5-7B<br>10 [1] 10 [2] 10 [1] 10 [2]<br>Model Size (Billion Parameters, Log Scale) Model Size (Billion Parameters, Log Scale)<br>Full Set: Task-wise Performance Hard Set: Task-wise Performance<br>50<br>60 Textual Reasoning Textual Reasoning<br>Code Reasoning 40 Code Reasoning<br>KG Reasoning KG Reasoning<br>40 Logic Reasoning 30 Logic Reasoning<br>20<br>20 10<br>0<br>10 [1] 10 [2] 10 [1] 10 [2]<br>Model Size (Billion Parameters, Log Scale) Model Size (Billion Parameters, Log Scale)<br>Figure 5: Reasoning performance for different model sizes.<br>30<br>20<br>R-LLM<br>10 NR-LLM<br>R-LLM Fit<br>NR-LLM Fit<br>0 Overall Fit<br>200 400 600 800 1000 1200 1400<br>Average Number of tokens in output of the LLM<br>Score (%) Score (%)<br>Score (%) Score (%)<br>Score (%)<br>**----- End of picture text -----**<br>


Figure 6: Model performance under different output lengths. 

## **4.5 Performance across Varing LLM Sizes** 

Experiments on our ONEEVAL benchmark, designed to assess diverse reasoning capabilities, confirm established scaling laws [1]: LLM performance consistently improves with increased model size across both a Full set of tasks and a dedicated Hard subset. While larger models demonstrate significantly better overall reasoning scores, the Hard Set effectively probes the limits of current models, yielding substantially lower absolute scores and highlighting the inherent difficulty of these challenging reasoning instances. The trend suggests a potential plateau in performance for the very largest models evaluated, indicating diminishing returns on parameter scaling for these specific reasoning modalities with current architectures and training methodologies. 

In the KG task, as the overall model size increases, the performance gap between models of the same size remains relatively unchanged. Conversely, in code-related tasks, this gap between same-sized models appears to widen as size increases. It suggests that code tasks require more sensitive, emergent logical reasoning capabilities highly dependent on subtle model variations at scale, thus widening the performance gap between same-sized models. Conversely, KG tasks primarily leverage scaling for robust, structured knowledge processing, where within-size differences are more stable. 

Most notably, Logic Reasoning shows limited positive scaling on the Full Set but demonstrates the most pronounced improvement with scale on the Hard Set. This suggests that larger LLMs possess enhanced logical processing capabilities that are primarily leveraged and become evident only when confronted with complex logical problems. The ONEEVAL benchmark thus effectively differentiates models based on their size and architecture, providing critical insights into how various reasoning skills evolve with scale and identifying areas where current models still face significant limitations. 

## **4.6 Impact of LLM Thinking Length** 

Based on Figure 6, which plots LLM performance against output length (serving as a proxy for reasoning chain length), a clear distinction emerges between Reasoning-LLMs (R-LLM) and Non-Reasoning 

8 

LLMs (NR-LLM). The fitted curve for R-LLMs exhibits a non-monotonic trend: performance initially increases with output length, reaching a peak at a moderate length, but then undergoes a significant decline as output length increases further. This empirical observation strongly supports the **Insight 3: Beyond a moderate length, the introduction of noise and potential errors outweighs the benefits of additional steps.** 

The fitted curve for R-LLMs (blue line) exhibits a prominent non-monotonic relationship. Performance initially increases as output length grows from short chains (e.g., <400 tokens), indicating that some level of explicit reasoning is beneficial. The performance peaks at a moderate output length, roughly between 800 and 1000 tokens. Crucially, as the output length increases beyond this optimal point, the fitted curve shows a sharp decline in performance. For the longest output lengths depicted (>1200 tokens), the predicted performance drops significantly, even falling into negative score values. 

The trend in R-LLMs suggests that while explicit reasoning steps are initially beneficial, there is an optimal chain length beyond which the accumulation of noise and potential errors with each additional step outweighs any potential gains. Unlike NR-LLMs, which show a more stable or gradually declining performance with length, R-LLMs demonstrate a vulnerability to excessive chain length, leading to performance degradation, which highlights the critical need for chain-length sensitivity and control mechanisms in complex reasoning tasks. 

## **5 Related Work** 

**LLM Benchmarks.** Numerous benchmarks and leaderboards have explored a wide range of topics and tasks for evaluating diverse LLM capabilities. Common LLM benchmarks can be categorized into four types: (1) Knowledge Evaluation benchmarks test the LLM’s mastery of subject knowledge through objective multiple-choice and open-ended questions, including MMLU [6], CMMLU [25], CEval [26], AGIEval [7]. (2) Instruction Following benchmarks like LLMBAR [27], Flan [28], and NaturalInstructions [29], evaluate the LLM’s ability to follow user instructions through QA formats. (3) Chat and Dialogue benchmarks focus on the LLM’s contextual understanding and conversational abilities using multi-turn interactive question-answering data, including CoQA [30], MMDialog [31], MT-Bench [8], and OpenAssistant [9]. (4) Safety and Risk benchmarks evaluate the LLM’s safety and hallucination risks through multiple-choice and open-ended questions, such as DecodingTrust [32], AdvGLUE [33], StrongReject [34], and HarmBench [35]. 

**LLM Leaderboards.** Unlike single evaluation benchmarks, the leaderboards integrate multiple evaluation tasks and consolidate a wide range of evaluation scenarios, adopting multi-dimensional and multi-task evaluation methods to construct a more comprehensive capability profile of LLMs. HuggingFace’s OpenLLM[1] Leaderboard combines six benchmarks to comprehensively diagnosis LLMs’ abilities in knowledge understanding, reasoning, mathematical problem-solving, information extraction, and complex task handling. Shanghai AI Lab’s OpenCompass [36] establishes an evaluation framework spanning six dimensions: examinations, knowledge, language, comprehension, reasoning, and safety, while incorporating dynamically updated evaluation datasets to ensure timeliness and breadth. FlagEval[2] integrates over 30 capability dimensions and 30 benchmarks, constructing a large-scale evaluation repository with more than 100,000 test samples, covering multimodal information such as text, images, and audio. In addition to standardized dataset-based evaluations, comparative evaluations based on human preferences have gained significant attention in recent years. UC Berkeley’s Chatbot Arena Leaderboard [37] innovatively employs a crowdsourcing + Elo ranking mechanism, allowing human evaluators to pair-wise compare the quality of LLMs’ responses, accumulating relative strength rankings among models. Stanford University’s AlpacaEval [38] introduces the “LLM-as-a-Judge” approach, leveraging large LLMs as judges to compare the responses of the evaluated model with those of a reference model, using relative win rates as the basis for ranking, significantly enhancing the scale and efficiency of evaluation. 

However, existing benchmarks and leaderboards still have several limitations: (1) lack of evaluation for reasoning capabilities based on complex knowledge bases; and (2) homogeneous knowledge carriers, failing to comprehensively cover evaluating dimensions such as knowledge graphs, code, and structured tabular knowledge. To address these issues, ONEEVAL introduces the first unified evaluation framework for cross-knowledge-source, multi-domain complex knowledge base reasoning 

> 1https://huggingface.co/open-llm-leaderboard 

> 2https://flageval.baai.ac.cn 

9 

tasks, aiming to achieve broader coverage and greater scientific rigor in evaluating the knowledgeenhanced capabilities of LLMs. 

## **6 Conclusion** 

In this work, we introduced ONEEVAL, a benchmark designed to evaluate LLMs on reasoning tasks involving structured external knowledge across various modalities and domains. Our evaluation of multiple state-of-the-art LLMs revealed significant limitations in structured reasoning, particularly as structural complexity increased. The models struggled with processing and reasoning over highly structured representations, and extending reasoning chains often led to diminishing returns. By releasing the ONEEVAL dataset, evaluation scripts, and baseline results, we aim to provide a platform for advancing research in this area. While ONEEVAL offers a diverse evaluation suite, its focus on static datasets may not fully capture the dynamic nature of real-world reasoning tasks, suggesting a direction for future improvements. Overall, our findings highlight the need for novel architectures, training paradigms, and reasoning strategies tailored to structured knowledge and formal systems. 

## **References** 

- [1] OpenAI. GPT-4 technical report. _CoRR_ , abs/2303.08774, 2023. 

- [2] Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky, Aiden Low, Alec Helyar, Aleksander Madry, Alex Beutel, Alex Carney, Alex Iftimie, Alex Karpenko, Alex Tachard Passos, Alexander Neitz, Alexander Prokofiev, Alexander Wei, Allison Tam, Ally Bennett, Ananya Kumar, Andre Saraiva, Andrea Vallone, Andrew Duberstein, Andrew Kondrich, Andrey Mishchenko, Andy Applebaum, Angela Jiang, Ashvin Nair, Barret Zoph, Behrooz Ghorbani, Ben Rossen, Benjamin Sokolowsky, Boaz Barak, Bob McGrew, Borys Minaiev, Botao Hao, Bowen Baker, Brandon Houghton, Brandon McKinzie, Brydon Eastman, Camillo Lugaresi, Cary Bassin, Cary Hudson, Chak Ming Li, Charles de Bourcy, Chelsea Voss, Chen Shen, Chong Zhang, Chris Koch, Chris Orsinger, Christopher Hesse, Claudia Fischer, Clive Chan, Dan Roberts, Daniel Kappler, Daniel Levy, Daniel Selsam, David Dohan, David Farhi, David Mely, David Robinson, Dimitris Tsipras, Doug Li, Dragos Oprica, Eben Freeman, Eddie Zhang, Edmund Wong, Elizabeth Proehl, Enoch Cheung, Eric Mitchell, Eric Wallace, Erik Ritter, Evan Mays, Fan Wang, Felipe Petroski Such, Filippo Raso, Florencia Leoni, Foivos Tsimpourlas, Francis Song, Fred von Lohmann, Freddie Sulit, Geoff Salmon, Giambattista Parascandolo, Gildas Chabot, Grace Zhao, Greg Brockman, Guillaume Leclerc, Hadi Salman, Haiming Bao, Hao Sheng, Hart Andrin, Hessam Bagherinezhad, Hongyu Ren, Hunter Lightman, Hyung Won Chung, Ian Kivlichan, Ian O’Connell, Ian Osband, Ignasi Clavera Gilaberte, and Ilge Akkaya. Openai o1 system card. _CoRR_ , abs/2412.16720, 2024. 

- [3] DeepSeek-AI, Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, Damai Dai, Daya Guo, Dejian Yang, Deli Chen, Dongjie Ji, Erhang Li, Fangyun Lin, Fucong Dai, Fuli Luo, Guangbo Hao, Guanting Chen, Guowei Li, H. Zhang, Han Bao, Hanwei Xu, Haocheng Wang, Haowei Zhang, Honghui Ding, Huajian Xin, Huazuo Gao, Hui Li, Hui Qu, J. L. Cai, Jian Liang, Jianzhong Guo, Jiaqi Ni, Jiashi Li, Jiawei Wang, Jin Chen, Jingchang Chen, Jingyang Yuan, Junjie Qiu, Junlong Li, Junxiao Song, Kai Dong, Kai Hu, Kaige Gao, Kang Guan, Kexin Huang, Kuai Yu, Lean Wang, Lecong Zhang, Lei Xu, Leyi Xia, Liang Zhao, Litong Wang, Liyue Zhang, Meng Li, Miaojun Wang, Mingchuan Zhang, Minghua Zhang, Minghui Tang, Mingming Li, Ning Tian, Panpan Huang, Peiyi Wang, Peng Zhang, Qiancheng Wang, Qihao Zhu, Qinyu Chen, Qiushi Du, R. J. Chen, R. L. Jin, Ruiqi Ge, Ruisong Zhang, Ruizhe Pan, Runji Wang, Runxin Xu, Ruoyu Zhang, Ruyi Chen, S. S. Li, Shanghao Lu, Shangyan Zhou, Shanhuang Chen, Shaoqing Wu, Shengfeng Ye, Shengfeng Ye, Shirong Ma, Shiyu Wang, Shuang Zhou, Shuiping Yu, Shunfeng Zhou, Shuting Pan, T. Wang, Tao Yun, Tian Pei, Tianyu Sun, W. L. Xiao, and Wangding Zeng. Deepseek-v3 technical report. _CoRR_ , abs/2412.19437, 2024. 

- [4] Simon Ott, Konstantin Hebenstreit, Valentin Liévin, Christoffer Egeberg Hother, Milad Moradi, Maximilian Mayrhauser, Robert Praas, Ole Winther, and Matthias Samwald. Thoughtsource: A central hub for large language model reasoning data. _Scientific data_ , 10(1):528, 2023. 

- [5] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. _Advances in neural information processing systems_ , 35:24824–24837, 2022. 

- [6] Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. In _9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021_ . OpenReview.net, 2021. 

10 

- [7] Wanjun Zhong, Ruixiang Cui, Yiduo Guo, Yaobo Liang, Shuai Lu, Yanlin Wang, Amin Saied, Weizhu Chen, and Nan Duan. Agieval: A human-centric benchmark for evaluating foundation models. In Kevin Duh, Helena Gómez-Adorno, and Steven Bethard, editors, _Findings of the Association for Computational Linguistics: NAACL 2024, Mexico City, Mexico, June 16-21, 2024_ , pages 2299–2314. Association for Computational Linguistics, 2024. 

- [8] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. Judging llm-as-ajudge with mt-bench and chatbot arena. In Alice Oh, Tristan Naumann, Amir Globerson, Kate Saenko, Moritz Hardt, and Sergey Levine, editors, _Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023_ , 2023. 

- [9] Andreas Köpf, Yannic Kilcher, Dimitri von Rütte, Sotiris Anagnostidis, Zhi Rui Tam, Keith Stevens, Abdullah Barhoum, Duc Nguyen, Oliver Stanley, Richárd Nagyfi, Shahul ES, Sameer Suri, David Glushkov, Arnav Dantuluri, Andrew Maguire, Christoph Schuhmann, Huu Nguyen, and Alexander Mattick. Openassistant conversations - democratizing large language model alignment. In Alice Oh, Tristan Naumann, Amir Globerson, Kate Saenko, Moritz Hardt, and Sergey Levine, editors, _Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023_ , 2023. 

- [10] Sébastien Bubeck, Varun Chandrasekaran, Ronen Eldan, Johannes Gehrke, Eric Horvitz, Ece Kamar, Peter Lee, Yin Tat Lee, Yuanzhi Li, Scott Lundberg, et al. Sparks of artificial general intelligence: Early experiments with gpt-4. _arXiv preprint arXiv:2303.12712_ , 2023. 

- [11] Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William Cohen, Ruslan Salakhutdinov, and Christopher D Manning. Hotpotqa: A dataset for diverse, explainable multi-hop question answering. In _Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing_ , pages 2369–2380, 2018. 

- [12] Mor Geva, Daniel Khashabi, Elad Segal, Tushar Khot, Dan Roth, and Jonathan Berant. Did aristotle use a laptop? a question answering benchmark with implicit reasoning strategies. _Transactions of the Association for Computational Linguistics_ , 9:346–361, 2021. 

- [13] DeepSeek-AI, Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, Xiaokang Zhang, Xingkai Yu, Yu Wu, Z. F. Wu, Zhibin Gou, Zhihong Shao, Zhuoshu Li, Ziyi Gao, Aixin Liu, Bing Xue, Bingxuan Wang, Bochao Wu, Bei Feng, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, Damai Dai, Deli Chen, Dongjie Ji, Erhang Li, Fangyun Lin, Fucong Dai, Fuli Luo, Guangbo Hao, Guanting Chen, Guowei Li, H. Zhang, Han Bao, Hanwei Xu, Haocheng Wang, Honghui Ding, Huajian Xin, Huazuo Gao, Hui Qu, Hui Li, Jianzhong Guo, Jiashi Li, Jiawei Wang, Jingchang Chen, Jingyang Yuan, Junjie Qiu, Junlong Li, J. L. Cai, Jiaqi Ni, Jian Liang, Jin Chen, Kai Dong, Kai Hu, Kaige Gao, Kang Guan, Kexin Huang, Kuai Yu, Lean Wang, Lecong Zhang, Liang Zhao, Litong Wang, Liyue Zhang, Lei Xu, Leyi Xia, Mingchuan Zhang, Minghua Zhang, Minghui Tang, Meng Li, Miaojun Wang, Mingming Li, Ning Tian, Panpan Huang, Peng Zhang, Qiancheng Wang, Qinyu Chen, Qiushi Du, Ruiqi Ge, Ruisong Zhang, Ruizhe Pan, Runji Wang, R. J. Chen, R. L. Jin, Ruyi Chen, Shanghao Lu, Shangyan Zhou, Shanhuang Chen, Shengfeng Ye, Shiyu Wang, Shuiping Yu, Shunfeng Zhou, Shuting Pan, and S. S. Li. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. _CoRR_ , abs/2501.12948, 2025. 

- [14] xAI. Grok 3 beta — the age of reasoning agents, February 2025. Accessed: 2025-05-15. 

- [15] OpenAI. Openai o3 and o4-mini system card. Technical report, OpenAI, April 2025. Accessed: 2025-0515. 

- [16] Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, Anirudh Goyal, Anthony Hartshorn, Aobo Yang, Archi Mitra, Archie Sravankumar, Artem Korenev, Arthur Hinsvark, Arun Rao, Aston Zhang, Aurélien Rodriguez, Austen Gregerson, Ava Spataru, Baptiste Rozière, Bethany Biron, Binh Tang, Bobbie Chern, Charlotte Caucheteux, Chaya Nayak, Chloe Bi, Chris Marra, Chris McConnell, Christian Keller, Christophe Touret, Chunyang Wu, Corinne Wong, Cristian Canton Ferrer, Cyrus Nikolaidis, Damien Allonsius, Daniel Song, Danielle Pintz, Danny Livshits, David Esiobu, Dhruv Choudhary, Dhruv Mahajan, Diego GarciaOlano, Diego Perino, Dieuwke Hupkes, Egor Lakomkin, Ehab AlBadawy, Elina Lobanova, Emily Dinan, Eric Michael Smith, Filip Radenovic, Frank Zhang, Gabriel Synnaeve, Gabrielle Lee, Georgia Lewis Anderson, Graeme Nail, Grégoire Mialon, Guan Pang, Guillem Cucurell, Hailey Nguyen, Hannah Korevaar, Hu Xu, Hugo Touvron, Iliyan Zarov, Imanol Arrieta Ibarra, Isabel M. Kloumann, Ishan Misra, Ivan Evtimov, Jade Copet, Jaewon Lee, Jan Geffert, Jana Vranes, Jason Park, Jay Mahadeokar, Jeet Shah, Jelmer van der Linde, Jennifer Billock, Jenny Hong, Jenya Lee, Jeremy Fu, Jianfeng Chi, Jianyu Huang, Jiawen Liu, Jie 

11 

Wang, Jiecao Yu, Joanna Bitton, Joe Spisak, Jongsoo Park, Joseph Rocca, Joshua Johnstun, Joshua Saxe, Junteng Jia, Kalyan Vasuden Alwala, Kartikeya Upasani, Kate Plawiak, Ke Li, Kenneth Heafield, Kevin Stone, and et al. The llama 3 herd of models. _CoRR_ , abs/2407.21783, 2024. 

- [17] Aohan Zeng, Bin Xu, Bowen Wang, Chenhui Zhang, Da Yin, Diego Rojas, Guanyu Feng, Hanlin Zhao, Hanyu Lai, Hao Yu, Hongning Wang, Jiadai Sun, Jiajie Zhang, Jiale Cheng, Jiayi Gui, Jie Tang, Jing Zhang, Juanzi Li, Lei Zhao, Lindong Wu, Lucen Zhong, Mingdao Liu, Minlie Huang, Peng Zhang, Qinkai Zheng, Rui Lu, Shuaiqi Duan, Shudan Zhang, Shulin Cao, Shuxun Yang, Weng Lam Tam, Wenyi Zhao, Xiao Liu, Xiao Xia, Xiaohan Zhang, Xiaotao Gu, Xin Lv, Xinghan Liu, Xinyi Liu, Xinyue Yang, Xixuan Song, Xunkai Zhang, Yifan An, Yifan Xu, Yilin Niu, Yuantao Yang, Yueyan Li, Yushi Bai, Yuxiao Dong, Zehan Qi, Zhaoyu Wang, Zhen Yang, Zhengxiao Du, Zhenyu Hou, and Zihan Wang. Chatglm: A family of large language models from GLM-130B to GLM-4 all tools. _CoRR_ , abs/2406.12793, 2024. 

- [18] An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. Qwen2.5 technical report. _CoRR_ , abs/2412.15115, 2024. 

- [19] Qwen Team. Qwq-32b: Embracing the power of reinforcement learning, March 2025. Accessed: 2025-0515. 

- [20] Meta AI. The llama 4 herd: The beginning of a new era of natively multimodal ai innovation, April 2025. Accessed: 2025-05-15. 

- [21] OpenAI. Introducing gpt-4.1 in the api, April 2025. Accessed: 2025-05-15. 

- [22] ByteDance. Doubao-1.5 pro, January 2025. 

- [23] Anthropic. Claude 3.7 sonnet, June 2024. Accessed: 2025-05-15. 

- [24] Google DeepMind. Gemini 2.5: Our most intelligent ai model, March 2025. Accessed: 2025-05-15. 

- [25] Haonan Li, Yixuan Zhang, Fajri Koto, Yifei Yang, Hai Zhao, Yeyun Gong, Nan Duan, and Timothy Baldwin. CMMLU: measuring massive multitask language understanding in chinese. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar, editors, _Findings of the Association for Computational Linguistics, ACL 2024, Bangkok, Thailand and virtual meeting, August 11-16, 2024_ , pages 11260–11285. Association for Computational Linguistics, 2024. 

- [26] Yuzhen Huang, Yuzhuo Bai, Zhihao Zhu, Junlei Zhang, Jinghan Zhang, Tangjun Su, Junteng Liu, Chuancheng Lv, Yikai Zhang, Jiayi Lei, Yao Fu, Maosong Sun, and Junxian He. C-eval: A multilevel multi-discipline chinese evaluation suite for foundation models. In Alice Oh, Tristan Naumann, Amir Globerson, Kate Saenko, Moritz Hardt, and Sergey Levine, editors, _Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023_ , 2023. 

- [27] Zhiyuan Zeng, Jiatong Yu, Tianyu Gao, Yu Meng, Tanya Goyal, and Danqi Chen. Evaluating large language models at evaluating instruction following. In _The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024_ . OpenReview.net, 2024. 

- [28] Shayne Longpre, Le Hou, Tu Vu, Albert Webson, Hyung Won Chung, Yi Tay, Denny Zhou, Quoc V. Le, Barret Zoph, Jason Wei, and Adam Roberts. The flan collection: Designing data and methods for effective instruction tuning. In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett, editors, _International Conference on Machine Learning, ICML 2023, 23-29 July 2023, Honolulu, Hawaii, USA_ , volume 202 of _Proceedings of Machine Learning Research_ , pages 22631–22648. PMLR, 2023. 

- [29] Swaroop Mishra, Daniel Khashabi, Chitta Baral, and Hannaneh Hajishirzi. Cross-task generalization via natural language crowdsourcing instructions. In Smaranda Muresan, Preslav Nakov, and Aline Villavicencio, editors, _Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2022, Dublin, Ireland, May 22-27, 2022_ , pages 3470–3487. Association for Computational Linguistics, 2022. 

- [30] Siva Reddy, Danqi Chen, and Christopher D. Manning. Coqa: A conversational question answering challenge. _Trans. Assoc. Comput. Linguistics_ , 7:249–266, 2019. 

12 

- [31] Jiazhan Feng, Qingfeng Sun, Can Xu, Pu Zhao, Yaming Yang, Chongyang Tao, Dongyan Zhao, and Qingwei Lin. Mmdialog: A large-scale multi-turn dialogue dataset towards multi-modal open-domain conversation. In Anna Rogers, Jordan L. Boyd-Graber, and Naoaki Okazaki, editors, _Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2023, Toronto, Canada, July 9-14, 2023_ , pages 7348–7363. Association for Computational Linguistics, 2023. 

- [32] Boxin Wang, Weixin Chen, Hengzhi Pei, Chulin Xie, Mintong Kang, Chenhui Zhang, Chejian Xu, Zidi Xiong, Ritik Dutta, Rylan Schaeffer, Sang T. Truong, Simran Arora, Mantas Mazeika, Dan Hendrycks, Zinan Lin, Yu Cheng, Sanmi Koyejo, Dawn Song, and Bo Li. Decodingtrust: A comprehensive assessment of trustworthiness in GPT models. In Alice Oh, Tristan Naumann, Amir Globerson, Kate Saenko, Moritz Hardt, and Sergey Levine, editors, _Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023_ , 2023. 

- [33] Boxin Wang, Chejian Xu, Shuohang Wang, Zhe Gan, Yu Cheng, Jianfeng Gao, Ahmed Hassan Awadallah, and Bo Li. Adversarial GLUE: A multi-task benchmark for robustness evaluation of language models. In Joaquin Vanschoren and Sai-Kit Yeung, editors, _Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks 1, NeurIPS Datasets and Benchmarks 2021, December 2021, virtual_ , 2021. 

- [34] Alexandra Souly, Qingyuan Lu, Dillon Bowen, Tu Trinh, Elvis Hsieh, Sana Pandey, Pieter Abbeel, Justin Svegliato, Scott Emmons, Olivia Watkins, and Sam Toyer. A strongreject for empty jailbreaks. In Amir Globersons, Lester Mackey, Danielle Belgrave, Angela Fan, Ulrich Paquet, Jakub M. Tomczak, and Cheng Zhang, editors, _Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15, 2024_ , 2024. 

- [35] Mantas Mazeika, Long Phan, Xuwang Yin, Andy Zou, Zifan Wang, Norman Mu, Elham Sakhaee, Nathaniel Li, Steven Basart, Bo Li, David A. Forsyth, and Dan Hendrycks. Harmbench: A standardized evaluation framework for automated red teaming and robust refusal. In _Forty-first International Conference on Machine Learning, ICML 2024, Vienna, Austria, July 21-27, 2024_ . OpenReview.net, 2024. 

- [36] OpenCompass Contributors. Opencompass: A universal evaluation platform for foundation models. `https://github.com/open-compass/opencompass` , 2023. 

- [37] Wei-Lin Chiang, Lianmin Zheng, Ying Sheng, Anastasios Nikolas Angelopoulos, Tianle Li, Dacheng Li, Banghua Zhu, Hao Zhang, Michael I. Jordan, Joseph E. Gonzalez, and Ion Stoica. Chatbot arena: An open platform for evaluating llms by human preference. In _Forty-first International Conference on Machine Learning, ICML 2024, Vienna, Austria, July 21-27, 2024_ . OpenReview.net, 2024. 

- [38] Yann Dubois, Balázs Galambosi, Percy Liang, and Tatsunori B. Hashimoto. Length-controlled alpacaeval: A simple way to debias automatic evaluators. _CoRR_ , abs/2404.04475, 2024. 

13 

## **A Details of Task Construction** 

## **A.1 BioTextQA** 

In the pursuit of assessing the understanding and reasoning capabilities of large language models when applied to highly specialized scientific domains, we construct three QA datasets: BioTextQA, MatTextQA, and PharmKGQA. These datasets are designed to challenge models’ understanding of biological sciences, material sciences, and pharmaceutical knowledge, respectively. The data sources for construction are derived from peer-reviewed publications and structured knowledge graphs within these domains: BioTextQA based on 3,881 biological papers from PubMed Central[3] , MatTextQA based on 8,222 materials science papers from arXiv[4] , PharmKGQA based on the PharmKG[5] containing 1,093,237 pharmaceutical relational triples. The workflow for dataset construction is consistent across all three domains, summarized in three sequential steps as follows: 

_**Step 1. Knowledge Source Selection.**_ Random samples of knowledge sources are extracted: for text-based domains like PubMed Central and materials science papers from arXiv, we randomly select entire documents or specific excerpts from the corpus. For PharmKG, relational triples conforming to predefined patterns are extracted. 

_**Step 2. Question Generation.**_ The process for generating QA pairs differs slightly based on the type of knowledge source: text-based Knowledge Sources: We input the extracted textual content into GPT-4o, instructing it to generate questions directly related to the given text. This ensures that the questions are grounded in the provided knowledge source. KG-Based Sources: Questions are generated using predefined relational pattern templates, guaranteeing balanced distribution of question types and ensuring precise correspondence between questions and answers. Subsequently, GPT-4o is utilized to refine the questions, enhancing their semantic richness and fluency. 

_**Step 3. Question Validation.**_ We employ GPT-4o for rigorous validation of generated QA pairs, focusing on two aspects: (1) meaningfulness of the question: The LLM assesses whether the question is coherent, contextually relevant, and scientifically valid;(2) answerability from the original knowledge source: LLM needs to verify whether the answer can be accurately traced back to the input text or knowledge graph. Questions must meet both requirements (both evaluated as “True”) to be retained in the dataset. Through this systematic workflow, we ensure that the constructed datasets are of high quality—containing meaningful, relevant, and source-grounded QA pairs. These datasets serve as benchmarks for evaluating LLMs’ performance under the demanding criteria of scientific knowledge understanding and reasoning. 

## **A.2 MatTextQA** 

The construction process of the MatTextQA dataset refers to the section A.1. 

## **A.3 ChineseLawFact** 

ChineseLawFact is a fact-checking task that focuses on the Chinese legal domain and aims to verify the accuracy of legal statements. It consists of 9,464 annotated legal statements. Statements were extracted from objective questions in the Chinese National Judicial Examination, with corresponding explanations sourced from official exam preparation materials. The task requires LLMs to possess not only deep knowledge of Chinese law, but also rigorous legal reasoning. 

To construct a high-quality fact-checking dataset, we utilized information from objective legal exam questions, including multiple-choice questions, their options, explanatory analyses, and associated legal knowledge points. When a question is deemed suitable for legal fact-checking tasks, we treat the question stem and each of the four answer options as four distinct factual claims. The accompanying explanatory analysis often contains legal provisions or knowledge that can serve as a basis for evaluating these claims—analogous to major premises in legal reasoning and these are extracted separately as part of the claim context. The remaining analysis can be divided into four segments, each corresponding to one of the answer options, providing correctness labels and detailed 

> 3https://pmc.ncbi.nlm.nih.gov/ 

> 4https://arxiv.org/ 

> 5https://github.com/biomed-AI/PharmKG 

14 

justifications. Each explanation serves as a minor premise in the structure of Chinese syllogistic legal reasoning, while the label represents the conclusion. 

## **A.4 AttributionNLI** 

AttributionNLI is an automatically constructed benchmark designed to evaluate the fine-grained and logically complex attribution capabilities of Attributed Question Answering (AQA) systems. CAQA introduces a comprehensive attribution taxonomy, consisting of four categories: supportive, partially supportive, contradictory, and irrelevant. Moreover, it distinguishes between different levels of reasoning complexity in attribution: single, union, intersection, and concatenation. These dimensions allow CAQA to test a wide spectrum of attribution behaviors, from basic evidence recognition to complex multi-hop reasoning. 

The construction of CAQA follows a four-step automated pipeline, grounded in knowledge graph (KG) semantics and logical query manipulation. The process is designed to scale easily across domains and supports the generation of high-quality attribution labels without manual intervention. (1) Query Collection. We begin by collecting natural language questions, answer entities, and logical queries from existing KGQA datasets such as GrailQA and WebQuestionsSP. These logical queries, often expressed in SPARQL-like syntax, are aligned with Freebase, a structured knowledge graph comprising relational triples in the form (h, r, t). (2) Query Extension. To simulate varying reasoning complexities, we apply logical operators to extend the original KG queries. Specifically, union and intersection operations are used to generate more complex query forms. These extensions transform single-triple, path-like, and tree-like queries into union-tree or intersection structures, which are associated with different complexity levels in attribution. (3) Structured Attribution Generation. Each extended query is grounded on the KG to extract a supporting subgraph. This subgraph is then programmatically edited to simulate other attribution categories: Supportive: The original subgraph as retrieved. Partially supportive: Derived by randomly deleting key triples from the subgraph. Contradictory: Constructed by replacing the answer entity with a type-consistent but incorrect entity. Irrelevant: Composed of unrelated triples while preserving structural similarity. (4) Data Generation. Using ChatGPT, we convert each subgraph into a natural language citation and paraphrase the question and answer accordingly. Each instance in CAQA thus contains a question, an answer statement, one or more citations, a categorical attribution label, and a complexity level. 

## **A.5 KCQAD** 

Knowledge Conflict Question Answering Dataset is a novel resource comprising 500 carefully constructed questions designed to evaluate a model’s ability to reconcile conflicting and evolving information. KCQAD captures both knowledge conflict and non-conflict scenarios, simulating real-world challenges where LLMs must navigate between their internal (parametric) knowledge and external (contextual) evidence to produce accurate answers. These scenarios are particularly relevant in dynamic domains such as current events, scientific discovery, and socio-political discourse, where information may change over time or differ across sources. 

To systematically capture knowledge change, we focus on two primary phenomena: perpetuation and evolution. These subsets of KCQAD are automatically curated using structured temporal data extracted from Wikipedia tables and Wikidata triples. Specifically, we identify facts associated with time-indexed entities (e.g., political leaders, event outcomes, organizational status) and generate question-answer pairs reflecting different temporal states. For each pair, we retrieve context passages from the corresponding Wikipedia snapshots to provide supporting evidence. This design ensures that the dataset aligns with the natural evolution of knowledge, facilitating future extensions as new data becomes available. 

In addition to temporal conflicts, KCQAD also incorporates misinformation-induced conflicts, where contradictions arise not from temporal updates but from deliberately incorrect information. To simulate these cases, we construct misinformation contexts by modifying original Wikipedia passages. This is done by replacing entities or facts with similar but incorrect alternatives in a way that maintains grammatical and semantic plausibility. These examples challenge models to detect and reason about factual inconsistencies that may not be apparent on the surface, closely mirroring the challenges posed by misinformation and fake news in real-world applications. 

15 

## **A.6 PharmKGQA** 

The construction process of the PharmKGQA dataset refers to the section A.1. 

## **A.7 AffairQA** 

This task is formulated such that the input is a specific government affairs question _q_ , and the output is a entity _e_ extracted from the government affair KG _K_ as the answer of question. Its construction process can be divided into the following three steps: 

_**Step1. Triple Extraction.**_ The original government data is sourced from publicly available csv files on the Zhejiang Province data open platform[6] . We extract triples from csv files covering 5 different government scenarios to construct the government affairs KG _K_ . The document topics cover various topics, including city history, rescue experts, libraries, medical institutions, and gas stations information. 

_**Step2. Data Cleaning and Validation.**_ At this stage, we manually review whether there are any missing elements in these triples and filter out the unqualified ones. The final government affairs knowledge graph comprises 470,631 triples. 

_**Step3. Question Generation.**_ We upload the triples corresponding to each topic to the official Qwen website[7] , prompting Qwen to randomly sample portions of the triples and synthesize corresponding questions. Human experts then perform a secondary verification of the generated questions. The resulting questions cover a variety of relation types, including but not limited to: containment relations, attribute relations, and spatiotemporal relations. 

## **A.8 PeopleRelQA** 

People are among the core elements of human social activities and events. Therefore, information retrieval and reasoning-based question answering about people are very common. Since factual information related to individuals is typically stored and represented using structured knowledge graphs, we have constructed a complex people relationship-centric KGQA dataset PeopleRelQA. 

For knowledge base construction, we extract 785,553 people relationship-related knowledge triples from infoboxes in Baidu Baike[8] entries to serve as the KG source. Based on this KG _K_ , we design question templates based on three types of multi-hop relational patterns: multi-hop query ( `A` _→_ `B` _→_ `C` _→_ `?` ), path checking ( `A` _→_ `B` _→_ `C` _→_ `D?` ), and compositional query ( `A` _→_ `B` _→_ `?` _←_ `D` ). 

Furthermore, we utilize the graph query language to retrieve all paths from the people relationship KG _K_ that fit these patterns and fill these paths into the question templates to obtain the initial question-answer pairs. To further increase question complexity, and ensure more natural expressions, we input these initial QA pairs into Qwen2-72B-Instruct for refinement, resulting in polished QA pairs involving multi-hop queries, counting tasks, logical compositions, and other complex KG reasoning questions. 

## **A.9 ReportFixer** 

Financial reports sometimes contain inconsistencies or errors, which may result from data processing mistakes, subjective oversight, or asynchronous information updates. In practical applications, automatically identifying and correcting these inconsistencies enhances the reliability and credibility of the reports. This dataset is designed to evaluate the capability of large language model systems in detecting discrepancies between triples and text within financial reports. Consequently, the construction process of the ReportFixer dataset is as follows: 

_**Step 1. Data Collection.**_ We search the CSCI Pengyuan website[9] using prefecture-level cities as keywords to find publicly available 2022 city investment company rating reports. In these financial reports, human experts manually select excerpts related to regional economic environments, including 

> 6https://data.zjzwfw.gov.cn/jdop_front/channal/data_public.do 

> 7https://chat.qwen.ai/ 

> 8https://baike.baidu.com/ 

> 9https://www.cspengyuan.com/ 

16 

details such as economic development level, fiscal status, and debt level. Based on these selected texts, human experts compare the content of the selected excerpts with internal regional data (in the form of triples) from the company’s reports. They also correct specific figures and remove data descriptions from the excerpts that are not present in the internal reports, thereby creating pairs of triples and text. 

_**Step 2. Hallucination Disturbance.**_ We employ GPT-4o-0513 to introduce hallucination disturbances in the textual segments, generating naturally misleading text along with specific disturbance evidence. 

_**Step 3. Manual Annotation.**_ Human experts use the specific disturbance evidence to compare the triples with the disturbed text, and identify all inconsistent triples. If the disturbance evidence in the text does not correspond to any triple facts, it is marked as none. 

Based on the above process, we establish the ReportFixer task, whose input includes a set of triples _K_ and a piece of disturbed text _D_ . The task requires identifying inconsistent triples within the input or confirming the consistency of the input information. 

## **A.10 VersiCode** 

Existing LLMs struggle with library version dynamics in code generation. Based on this reason, we curate VersiCode, which was constructed through a multi-stage process involving data collection from three primary sources: popular Python libraries’ source code (extracting API examples across versions), downstream application code from research papers (capturing real-world usage evolution), and Stack Overflow Q&A pairs mentioning specific library versions. The raw data underwent rigorous cleaning using heuristic rules and hybrid human-LLM annotation to create structured metadata, followed by API lifecycle tagging to identify additions, deprecations, and version-specific behaviors. The final dataset was organized into task-specific subsets with executable test cases, incorporating version-controlled environments and comprehensive evaluation metrics to assess model performance on version-aware code generation and migration tasks. 

## **Task Type** 

- Version-Specific Code Completion (VSCC) 

- Version-Aware Code Migration (VACM) 

## **Data Collection:** 

- Library Source Code: 300+ Python libraries (2,207 versions), API examples from docstrings 

- Downstream Applications: Code from top-tier papers & open-source projects 

- Stack Overflow: Version-annotated Q&A snippets 

**Data Structure:** Each instance is structured as: 

_m_ = �Library _l_ ; Version _v_ ; Description _d_ ; Code _c_ � 

- **Lifecycle Tags** : 

- Addition: API introduced at _vs_ , active in [ _vs, ve_ ) 

- Deprecation: API deprecated at _ve_ (last valid version) 

- **Granularities** : Token-level, line-level, & block-level masking 

**Migration Pairs:** Constructed as ( _mi, mj_ ) where: 

**==> picture [117 x 11] intentionally omitted <==**

covering both old _→_ new and new _→_ old migrations. 

## **Quality Control:** 

- Hybrid Annotation: GPT-4 auto-labeling + human validation 

- Metric: Critical Diff Check (CDC@k) for API usage, parameter matching, and syntax 

## **Statistics:** 

- 11k+ completion instances, 76k+ migration instances 

- Covers 9 years of library versions (2015–2023) 

17 

## **A.11 SymenTex-ASC** 

An ASP program is a set of rules of the following form: 

**==> picture [281 x 25] intentionally omitted <==**

where each _αi_ ( **x** _i_ ) is a literal of the form `p` ( **x** _i_ ) (positive literal) or `-p` ( **x** _i_ ) (negative literal), and each **x** _i_ consists of variables and constants. In ASP, “ `not` ” and “ `-` ” are called the default negation and the classical negation (strong negation), respectively. An ASP program or a rule is ground if there are no variables. A fact is a ground rule with _n_ = 0. We often write an ASP problem as a pair ( _W, D_ ) with _W_ a set of facts, and _D_ a set of rules. 

For example, assuming the bird is named Tweety, the three ASP programs _Pi_ = ( _Wi, D_ ) _, i_ = 0 _,_ 1 _,_ 2, where 

**==> picture [223 x 11] intentionally omitted <==**

**==> picture [157 x 11] intentionally omitted <==**

**==> picture [317 x 17] intentionally omitted <==**

**==> picture [215 x 11] intentionally omitted <==**

Initially, since _W_ 0 contains only “Bird( _Tweety_ )”, _P_ 0 intuitively entails “CanFly( _Tweety_ )”. The new information “Injured( _Tweety_ )” in ( _W_ 1 _, D_ ) triggers the second rule in _D_ , entails “Abnormal( _Tweety_ )”, and invalidates the first rule in _D_ . Finally the fact “SlightlyInjured( _Tweety_ )” in ( _W_ 2 _, D_ ) invalidates “Abnormal( _Tweety_ )”, allowing “CanFly( _Tweety_ )” to be inferred once again. 

The semantics of ASP are characterized by the notion of answer sets, also known as stable models. An answer set _S_ of ( _W_ , _D_ ) satisfies the following properties: 

(1) _W ⊆ S_ : All facts in _W_ are included in the answer set _S_ . 

(2) For every rule ( _ω ← α_ 1 _, . . . , αm,_ not _αm_ +1 _, ...,_ not _αn_ ) _∈ D_ , if _α_ 1 _, . . . , αm ∈ S_ and _αm_ +1 _, ..., αn ∈/ S_ , then _ω ∈ S_ . This ensures that the rules in _D_ are respected in _S_ . 

**Answer Set Computation (ASC):** Given an ASP program _P_ (which is guaranteed to have one or more answer sets, potentially generated using disjunction in rule heads), the task is to compute and return one of its correct answer sets. 

Formally, the task is to find a set _S[′]_ such that: 

**==> picture [224 x 11] intentionally omitted <==**

The output is such a set _S[′]_ . AS( _P_ ) denote the set of all answer sets of the program _P_ . 

## **B Input and Output Examples for Each Task** 

Table 3 shows the instruction and an example of BioTextQA. Table 4 shows the instruction and an example of MatTextQA. Table 5 shows the instruction and an example of ChineseLawFact. Table 6 shows the instruction and an example of AttributionNLI. Table 7 shows the instruction and an example of KCQAD. Table 8 shows the instruction and an example of PharmKGQA. Table 9 shows the instruction and an example of AffairQA. Table 10 shows the instruction and an example of PeopleRelQA. Table 11 shows the instruction and an example of ReportFixer. Table 12 shows the instruction and an example of VersiCode. Table 13 shows the instruction and an example of SymTex-ASC. 

18 

**INPUT:** Answer the question based on the document. 

Context: [’A tight cascade of gene regulation during the lifecycle of the malaria parasite in human blood cells suggests new functions for many Plasmodium genes’, ’Plasmodium chabaudi chabaudi is a malaria parasite of murine rodents. It has been widely used as a model to study various aspects of parasite biology and disease which are difficult to investigate using human malaria parasites. For instance, P. c. chabaudi is being used to study the genetic basis of drug resistance [1-4] and strain-specific immunity [5], because the execution and analysis of genetic crosses is relatively straightforward in this species [6]. The analysis of the genetic basis of aspects of malaria biology has been facilitated by recent developments in malaria genomics. Firstly, the Plasmodium falciparum genome has been fully sequenced and mapped [7] and there is also extensive sequence data now available for three of the four main malaria parasites of murine rodents [8]. Secondly, the degree of homology and conservation of gene synteny between the various species of malaria [4,9,10] allows the undertaking of comparative genomics and facilitates the elaboration of accurate genomic maps in these species.’, ’These data do not support the hypothesis of parasite growth inhibition due to an inhibition of the parasite SPM synthase activity as was demonstrated for PPMP [19-22] : 1) The anti-Plasmodium activity of AD2646 does not correlate with its inhibitory activity on the SPM synthase. Although AD2646 and PPMP showed similar inhibitory activity on this enzymic activity in parasites in cultures, AD2646 is about 300 times more efficient in inhibiting parasite development than PPMP; 2) In contrast to PPMP which inhibits the parasite development preferentially and reversibly at the ring stage [19], AD2646 inhibited parasite development preferentially and irreversibly at the trophozoite stage (Figure 5); 3) Inhibition of the SPM synthase activity by PPMP is associated with an inhibition of the TVN formation [19-22]. This was not observed in the presence of AD2646 (Figure 7).’, ’Figure 2 Monthly parasite and blood smear examination incidence patterns. Monthly parasite incidence patterns of P. falciparum and P. vivax malaria combined per 1000 population (red line on logarithmic scale), blood smears examined per 1000 population (black line on logarithmic scale), and percentage of blood smears positive for malaria (blue line) from January 1995 to October 2004 in Sri Lanka.’, ’Within-host competition in P. chabaudi is now firmly established [8,15,16]. Evidence for competition between co-infecting genotypes in human malaria infections is necessarily indirect, but consistent with this [4]. In older children and adults, for example, parasite densities do not increase with increasing numbers of clones, thus indicating that parasite clones are not regulated independently [17]. Given this, and the high frequency of mixed infections in human malaria [1-3,18] often consisting of both resistant and sensitive genotypes [19], and the fact that genetic diversity can be altered by antimalaria prophylaxis [20], it is highly likely that competitive release of drug resistance also occurs in human malaria. Indeed, a recent study has already implicated release of within-host competition as a key-factor in the spread of drug resistance in Uganda [21].’] Please use the context to answer the following question. You should give your answer in following format: [!Format Start!]A[!Format End!]. If there are multiple answers, please List all the answers divided with a comma in the same format. Question: what is the genomic activity of the malaria parasite in human blood cells. Options: A. Genomic Activity of the Parasite in Human Blood Cells B. Microarray Analysis: Genome-Scale Data Gathering C. Genomic Activity of the Parasite in Mosquito Cells D. Parasitic Interaction with Red Blood Cells Let’s think step by step! **OUTPUT:** A. Genomic Activity of the Parasite in Human Blood Cells 

Table 3: The instruction and an example of BioTextQA. 

**INPUT:** Context: [’ice phases, in contrast to data-intensive ML approaches that need the preparation of large trajectory datasets by running molecular dynamics simulations with carefully chosen force fields and/or prelabeling efforts of water phases. 2) In the model-free classification step, instead of applying hand-crafted order parameters or machine-learned latent features, we employ a universal atomic descriptor, the Smooth Overlap of Atomic Positions (SOAP), 20 which ensures Euclidean symmetries with translation and rotation invariance and transferability to any structural system with different symmetries. This approach allows us to distinguish ice phases in the test datasets with a remarkable 100% accuracy using only seven ideal reference structures of ice phases as model inputs. It demonstrates the generalizability of the score-based denoiser model in facilitating phase identification for complex molecular systems, and the unsupervised classification strategy proposed in this work can be generally applied to investigate structural evolution and phase identification for a variety of materials. Method Denoiser model As discussed in Ref., 19 the key component of the denoiser model is the noise prediction network _ϵθ_ ( _r_ ) that predicts the “noise” or displacement vectors of input atomic structure r with respect to reference structure r0. Such a model can be used to “denoise” or remove’, ’molecular dynamics 

19 

simulations, facilitating the subsequent classification tasks that identify phases by comparing the structural similarity 14 of denoised structures with ideal reference phases. The proposed approach achieves 100% accuracy for phase classification of seven ice phases without the need for training trajectories and/or label information from MD simulations. The training of the denoiser model requires only ideal reference structures. The simplicity and generalizability of this framework allow it to be transferable and facilitate phase identification for complex polymorph structures. Moreover, the successful application of our denoising and classification approach to iceliquid two-phase systems highlights its versatility in revealing the underlying crystalline structure of the ordered phases while preserving the liquid-like nature of the disordered phase. It showcases the possibility to enable a detailed analysis of thermodynamics and kinetics of the melting process, ice growth from super-cooled water, and the behavior of the ice/quasi-liquid interface at the atomic level without the need for prior phase labeling. This makes it a powerful tool for investigating complex systems where phase boundaries and transitions may not be well-defined or easily identifiable. In summary, our phase classification framework represents an important advancement in the accurate identification of ice phases in molecular dynamics simulations.’, ’The gain in accuracy is significant by following our approach. In Figure 4, we show the results for two different models. The first model is using the approach laid out in Figure 1 and described in this paper. This results in a model with R2 = 86.6%. To highlight the gain from our approach, we repeated the analysis following a traditional approach (labeled as “Processing as separate term”), where the annealing time and temperature were used as the two additional descriptors for the regression model, in addition to the parameters from the chemistry only network. Thus, time and temperature were used in place of the process descriptors discussed in Figure 3. This resulted in a model with the reduced accuracy of R2 = 69.1%. Therefore, this added step to the framework resulted in a significant accuracy and therefore allows us to reasonably screen the entire chemistry / processing design space rapidly. We can now predict the capacity value for systems that have not been experimentally measured (ie. ‘virtual’ materials). We can randomly generate combinations of new chemistries and process conditions, and then repeat the process. While there are no required constraints in defining these compositions;’] 

Please use the context to answer the following question. You should give your answer directly in following format: [!Format Start!]Your Answer[!Format End!]. If there are multiple answers, please List all the answers divided with a comma in the same format. 

Question: What is the the accuracy achieved for distinguishing ice phases using SOAP descriptor and ideal reference structures Options: A. 95% accuracy B. 98% accuracy C. 85% accuracy D. 100% accuracy Let’s think step by step! **OUTPUT:** C. 85% accuracy 

## Table 4: The instruction and an example of MatTextQA. 

**INPUT:** `你是一名中经验丰富的中文法律专家，擅长法律事实核查验证，现在有一个情节和相关的法 律声明，请根据专业知识判断其是否存在错误，并在最后输出结果` ‘ `正确` ‘ `或` ‘ `错误` ‘ `。` 1. `必要时，可以输出法条进行推理` 2. `提供详细的解释` 3. `一步步思考后给出结论` 4. `输出结果时请使用` ‘ `结果` ‘ `：` ‘ `正确` ‘ `或` ‘ `错误` ‘ `。` 5. `输出结果后，立即结束，不需要额外输出解释` ### `情节：` ‘ `村集体雇了专业公司甲公司开飞机洒农药，飞机飞得低，且途经乙养鸡场。后乙养鸡场 向丙履约，因为鸡的重量低于合同要求，损失` 10 `万元。乙养鸡场就认为是飞机把肉鸡吓得食欲下降饿 瘦了，乙养鸡场和甲公司协商无果，将甲公司诉至法院。` ‘ ### `法律声明：` ‘ `甲公司应当对没有因果关系承担责任` ‘ **OUTPUT:** `正确` 

## Table 5: The instruction and an example of ChineseLawFact. 

**INPUT:** Your task is to evaluate the relationship between a provided citation and the answer to a specific question. There are four possible types of relationships: 

Supportive: Choose this if the citation directly confirms or is fully in alignment with the answer, providing all necessary information to substantiate it. 

Insufficient: Choose this when the citation provides only partial backing for the answer, lacking some essential details or evidence needed for full support. 

Contradictory: Choose this option if the citation is consistent with the intent of the question but directly opposes or contradicts the answer. 

20 

Irrelevant: Select this option if the citation does not match the intent of the question and contains information that is not useful for answering. 

For each example provided: First, you need to look at the question given and the answer provided. Then, compare them with the content of the citation. Finally, select the appropriate relationship category based on whether the citation supports the answer, is missing information, contradicts itself, or is irrelevant to the answer. Example: Question: what cheese made from milk of dromedary camel has the same texture as affidelice au chablis does? Answer: Caravane cheese is the cheese made from the milk of dromedary camel that has the same texture as affidelice au chablis. Reference: Caravane cheese is a type of food cheese that has a soft texture. It belongs to the category of soft cheeses along with Affidelice au Chablis. **OUTPUT:** Relationship Category: Insufficient 

Table 6: The instruction and an example of AttributionNLI. 

**INPUT:** Answer the question based on the document. ### Document: On 27 February 2024, Legends Z-A was announced during a Pokémon Presents presentation with a release window of 2025. At the end of the trailer, the Mega Evolution mechanic, which was first introduced in X and Y, was also teased to return. Pokémon Legends: Z-A is being developed by Electronic Arts. Pokémon Legends: Z-A is an upcoming 2025 video game developed by Electronic Arts and published by Nintendo and The Pokémon Company for the Nintendo Switch. Announced in February 2024, Legends: Z-A is part of the ninth generation of Pokémon video games. It takes place in Lumiose City in the Kalos region, based on Paris, France, which originated in Pokémon X and Y (2013). Legends: Z-A is the second Pokémon Legends game, following Legends: Arceus (2022). The game will take place in the Kalos region, which is based on France and was first introduced in the 2013 video games Pokémon X and Y. According to its trailer, it is set during an urban redevelopment project at Lumiose City, which is based on Paris, with Nintendo of America stating that it will take place entirely within Lumiose City. ### Question: Who developed Pokémon Legends: Z-A? **OUTPUT:** Game Freak. 

Table 7: The instruction and an example of KCQAD. 

**INPUT:** Triplets: [ ‘p105rb,gene,H,retinoblastoma,disease,“1844252.0, 7787878.0, 9546379.0, 7698220.0, 1844252.0, 2562189.0, 12210730.0, 7787878.0”,“’Because the product of the retinoblastoma gene , p105Rb , is expressed in all cell types , the obvious question is what accounts for these tissue specific differences in the role of p105Rb .’, ‘Notably , both of the two giant regulators of checkpoint 1 -LRB- i.e. , p105RB -LSBretinoblastoma oncosuppressor-encoded protein -RSB- and p53 dependent WAF1/CIP1 -RRB- are influenced by or influence G1 cyclins : cyclin E/cdk2 kinase complexes hyperphosphorylate p105RB , induce E2F release , and free G1 exit .’, ‘The tumor suppressor genes p105RB -LRB- retinoblastoma , acting through the E2F transcription factor family -RRB- and p53 regulate cell proliferation , cell senescence , and apoptosis in many cell types .’, ‘Substrates for cyclin_D1 / Cdks have not been identified in vivo , but it has been proposed that the D class of cyclins might play a role in regulating the activity of the retinoblastoma gene product p105Rb .’, ‘Because the product of the retinoblastoma gene , p105Rb , is expressed in all cell types , the obvious question is what accounts for these tissue specific differences in the role of p105Rb .’, ‘Both the E1A and SV40 large T proteins contain similar CKII consensus sites proximal to the regions required for their associations with the retinoblastoma gene product -LRB- p105Rb -RRB- .’, ‘The retinoblastoma susceptibility gene product , p105Rb -LRB- RB -RRB- , is generally believed to be an important regulator in the control of cell growth , differentiation , and apoptosis .’, ‘Notably , both of the two giant regulators of checkpoint 1 -LRB- i.e. , p105RB -LSBretinoblastoma oncosuppressor-encoded protein -RSB- and p53 dependent WAF1/CIP1 -RRB- are influenced by or influence G1 cyclins : cyclin E/cdk2 kinase complexes hyperphosphorylate p105RB , induce E2F release , and free G1 exit .’ ”] 

Please use the triplets to answer the following question. You should give your answer in following format: [!Format Start!]Answer[!Format End!]. If there are multiple answers, please List all the answers divided with a comma in the same format. 

Question: Please answer the following question: what entities are connected to 1 25 dihydroxycholecalciferol through O? Let’s think step by step! 

**OUTPUT:** intercellular adhesion molecule 1 

Table 8: The instruction and an example of PharmKGQA. 

21 

**INPUT:** `根据以下三元组列表和您自己的知识背景，回答以下问题。在输出的最后一行，列出所有答 案。你的答案应该仅包含用逗号分隔的答案。` ### `三元组信息` : ( `永康历山省级森林公园，级别，省级），（永康历山省级森林公园，所在区 县，永康市），（永康历山省级森林公园，所在城市，金华市），（永康历山省级森林公园，所在 省，浙江省），（永嘉县五星潭省级森林公园，级别，省级），（永康历山省级森林公园，建造时 间，` 2016/1/1 `），（永嘉县五星潭省级森林公园，所在区县，永嘉县），（永嘉县五星潭省级森林公 园，所在省，浙江省），（永嘉县五星潭省级森林公园，所在城市，温州市），（遂昌县湖山森林 公园有限公司，所在省，浙江省），（永康历山省级森林公园，` type `，公园），（铜铃山国家森林公 园，地址，浙江省温州市文成县叶胜林场），（象山清风寨省级森林公园，级别，省级），（永嘉县 五星潭省级森林公园，建造时间，` 2008/1/1 `），（诸暨杭坞山森林公园，所在省，浙江省），（象山 南田岛省级森林公园，所在省，浙江省），（龙湾潭国家森林公园，所在区县，永嘉县），（牛头山 国家森林公园，所在省，浙江省），（玉苍山国家森林公园，所在省，浙江省），（铜铃山国家森林 公园，所在省，浙江省），` ### `问题` : `永康历山省级森林公园是在哪一年建造的？让我们一步一步思考！` 

**OUTPUT:** 2016/1/1 

Table 9: The instruction and an example of AffairQA task. 

**==> picture [410 x 367] intentionally omitted <==**

**----- Start of picture text -----**<br>
INPUT: 根据以下三元组列表和您自己的知识背景，回答以下问题。在输出的最后一行，列出所有答<br>案。你的答案应该仅包含用逗号分隔的答案。<br>###  三元组信息 : （胡适，主要作品，读梁漱溟先生的《东西文化及其哲学》），（拿什么拯救你，我<br>的爱人，主要角色，祝四萍），（拿什么拯救你，我的爱人，主要角色，罗晶晶），（如果高中棒球<br>的女经理人读过杜拉克的《管理学》的话，主要角色，朽木文明），（朱迪 · 福斯特，参演，魔幻迷<br>宫：制作《沉默的羔羊》），（斯戴芬 · 莫昌特，参演，《临时演员》第一季），（斯戴芬 · 莫昌特，<br>一<br>执导，《临时演员》第 季），（拿什么拯救你，我的爱人，主要演员，傅晶），（拿什么拯救你，<br>我的爱人，主要演员，刘烨），（拿什么拯救你，我的爱人，主要演员，张谦），（拿什么拯救你，<br>我的爱人，主要演员，于娜），（拿什么拯救你，我的爱人，主要角色，韩丁），（拿什么拯救你，<br>我的爱人，主要演员，姚岗），（拿什么拯救你，我的爱人，主要角色，程瑶），（拿什么拯救你，<br>我的爱人，主要角色，姚大维），（激流～你还记得我吗？～，主要角色，东耕司），（别担心，<br>他不会走远的，主要演员，奥莉维亚 · 汉密尔顿），（妈妈，不当你的女儿可以吗？，主要角色，松<br>岛太一），（维尼 · 琼斯，参演，《临时演员》第一季），（如果高中棒球的女经理人读过杜拉克的<br>《管理学》的话，主要角色，宫田夕纪），（如果高中棒球的女经理人读过杜拉克的《管理学》的<br>话，主要演员，岸南），（如果高中棒球的女经理人读过杜拉克的《管理学》的话，主要演员，铃<br>木裕树），（巴黎，我爱你，主要演员，凯特琳娜 · 桑迪诺 · 莫雷诺），（如果高中棒球的女经理人读<br>过杜拉克的《管理学》的话，主要角色，浅野庆一郎），（罗素 · 克劳，参演，《危情三日》的男人<br>们），（对不起，我爱你，主要演员，沃拉甘 · 罗娜瓦查拉），（妈妈，不当你的女儿可以吗？，主<br>要角色，早濑浩司），（别担心，他不会走远的，主要演员，凯瑞 · 布朗斯滕），（别担心，他不会<br>走远的，主要演员，杰昆 · 菲尼克斯），（如果高中棒球的女经理人读过杜拉克的《管理学》的话，<br>主要演员，大泉洋），（张雨生，音乐作品，爱的只是你（若我告诉你其实我爱的只是你）），（如<br>果高中棒球的女经理人读过杜拉克的《管理学》的话，主要演员，濑户康史），（浪客剑心：传说的<br>完结篇，主要角色，明神弥彦），（如果高中棒球的女经理人读过杜拉克的《管理学》的话，主要<br>角色，北条文乃），（如果高中棒球的女经理人读过杜拉克的《管理学》的话，主要演员，石冢英<br>彦），（如果高中棒球的女经理人读过杜拉克的《管理学》的话，主要角色，宫田靖代），（妈妈，<br>不当你的女儿可以吗？，主要演员，南波），（对不起，我爱你，主要演员，莫茶诺 · 欣彩萍翩），<br>（丁丁历险记：独角兽号的秘密，主要角色，阿道克船长），（如果高中棒球的女经理人读过杜拉克<br>的《管理学》的话，主要角色，川岛南），（如果高中棒球的女经理人读过杜拉克的《管理学》的<br>话，主要角色，星出纯），（别担心，他不会走远的，主要演员，杰克 · 布莱克），（妈妈，不当你<br>的女儿可以吗？，主要演员，麻生未），（拿什么拯救你，我的爱人，主要演员，李苒苒），（如果<br>高中棒球的女经理人读过杜拉克的《管理学》的话，主要演员，池松壮亮），（纽约，我爱你，主要<br>演员，卡洛斯 · 阿科斯塔），（激流～你还记得我吗？～，主要演员，南乃彩希），（如果高中棒球<br>的女经理人读过杜拉克的《管理学》的话，主要演员，松岛庄汰），（黄磊，主要作品，我的肩膀，<br>她们的翅膀），（别担心，他不会走远的，主要演员，鲁妮 · 玛拉）<br>###  问题 :  《利箭纵横》的主要演员的哪位搭档与陈道明搭档过？让我们一步一步思考！<br>OUTPUT: 王志文<br>**----- End of picture text -----**<br>


## Table 10: The instruction and an example of PeopleRelQA task. 

**INPUT:** `你是一位经济领域的专家，你将接收两个输入：` 1. `一组三元组，描述某个领域的事实。` 2. `一 段描述相同或相关领域的文本。你的任务是判断这两种输入中描述的事实是否存在冲突。` 

22 

`三元组信息：` “ `武进区` ”:[[2020, `综合财力` , 478.75 `亿元` ], [2021, `转移性收入` , 28.55 `亿元` ], [2020, `政府性基 金收入` , 247.86 `亿元` ], [2020, `转移性收入` , 35.69 `亿元` ], [2021, `税收占比` , 86.89%], [2021, `政府性基金收入` , 365.44 `亿元` ], [2021, `一般公共预算收入规模全市下辖区县中排名` , 1], [2021, `综合财力` , 610.55 `亿元` ], [2021, `税收收入` , 188.16 `亿元` ], [2021, `一般公共预算收入增速` , 10.93%], [2021, `财政自给率` , 103.42%], [2021, `一 般公共预算收入` , 216.55 `亿元` ], [2021, `一般公共预算支出` , 209.40 `亿元` ]] `文本：` 2021 `年武进区一般公共预算收入` 216.55 `亿元，较上年增长` 10.93% `，规模在常州市下辖区县中 排名第` 1 `位；其中税收收入` 188.16 `亿元，税收占比` 86.89% `；一般公共预算支出` 209.40 `亿元，财政自给 率` 103.42% `。政府性基金收入` 365.44 `亿元（上年同期为` 247.86 `亿元）；转移性收入` 28.55 `亿元（上年同 期为` 35.69 `亿元）；综合财力为` 610.55 `亿元（上年同期为` 478.75 `亿元）。 请找出与文本不一致的三元组，这些三元组用逗号分隔，如果没有，请回答无。` **OUTPUT:** `无` 

## Table 11: The instruction and an example of ReportFixer task. 

## **INPUT:** 

You are a professional Python engineer. I will provide functional descriptions and versions of specified dependency packages. You need to think step by step: 1. Understand the requirements 2. Consider how to use the dependencies 3. Implement the solution You need to write code in Python to implement this feature based on the functional description and using the dependency package and version I specified. Please note that you only need to return the code that implements the function, and do not return any other content. Please use `<start>` and `<end>` to enclose the generated code. Here is an example: 

**Function Description:** The function of this code is to print the results predicted by calling the model using vllm. **Dependency and version:** vllm==0.3.3 **Response:** `<start> for output in outputs: prompt = output.prompt generated_text = output.outputs[0].text print("Prompt,Generated text") <end>` **Dependency and version:** accelerate==0.20.0 **Functionality description of the code:** This code splits a list of elements between two processes and prints the inputs assigned to each process. The second split includes padding the inputs to make the assignment equal. **Refactored new code:** 

**OUTPUT:** 

23 

```
<start>
#Assumetherearetwoprocesses
fromaccelerateimportPartialState
state=PartialState()
withstate.split_between_processes(["A","B","C"])asinputs:
print(inputs)
#Process0
["A","B"]
#Process1
["C"]
withstate.split_between_processes(["A","B","C"],apply_padding=True)asinputs:
print(inputs)
#Process0
["A","B"]
#Process1
["C","C"]
<end>
```

Table 12: The instruction and an example of VersiCode Task. 

## **INPUT:** 

# Task: Generate a valid Answer Set (Stable Model) for the given ASP program. ## Understanding the Task You are given an Answer Set Programming (ASP) program consisting of facts and rules. Your task is to find *one* valid Answer Set (also known as a Stable Model) for this program. ## Key ASP Concepts Recap * **Facts:** Ground atoms assumed to be true (e.g., ‘p(a).‘). * **Rules:** Statements of the form ‘Head :- Body.‘ ("If ‘Body‘ is true, ‘Head‘ must be true"). * ‘Body‘ can contain positive literals (‘q(X)‘), strongly negated literals (‘-r(X)‘), and default-negated literals (‘not s(X)‘). * **Default Negation (‘not‘):** ‘not p‘ holds if ‘p‘ cannot be derived. * **Strong Negation (‘-‘):** ‘-p‘ means ‘p‘ is explicitly false. * **Answer Set (Stable Model):** A set of ground literals ‘A‘ that is: 1. **Consistent:** Does not contain ‘p‘ and ‘-p‘ simultaneously. 2. **Stable:** ‘A‘ is the minimal classical model of the program’s reduct ‘ _P[A]_ ‘ (formed by simplifying rules based on ‘A‘). Essentially, everything in ‘A‘ must be derivable from the simplified program, and nothing more. ## Goal Your goal is to output a single set of ground literals that constitutes a valid Answer Set for the program defined by ‘[facts]‘ and ‘[rules]‘. There might be multiple possible Answer Sets; you only need to provide one. 

24 

```
<start>
-
P17("Christopher").
-P18("John","Charles","Lucas").
-P19("John","Charles","Lucas").
-P20("John","Charles").
-P21("Barbara","John","Lucas").
-P24("Christopher","John","Charles").
-P28("Christopher","John","Charles").
-P30("Barbara","Christopher","John").
P12("Barbara","Christopher").
P13("Barbara","John","Charles").
P14("Barbara","John","Lucas").
P15("Christopher","John","Lucas").
P16("Christopher","Charles","Lucas").
P22("Barbara","John","Lucas").
P23("Barbara","John","Lucas").
P25("Christopher","Charles").
P26.
P31("Barbara","Christopher","Lucas").
P32("Barbara","Christopher","Lucas").
-P0(V1,V3):-P2(V1,V3,V4),notP5(V1,V3,V4),not-P16(V1,V3,V4).
-P1(V1):-P0(V1,V3),P5(V1,V3,V4),P25(V1,V3).
-P11(V1,V3,V4):--P0(V1,V3),-P4(V0,V1,V3),-P6(V0,V1,V4).
-P12(V0,V1):-P3(V0,V2,V4),P11(V1,V3,V4).
-P14(V0,V2,V4):-P0(V1,V3),P3(V0,V2,V4),P6(V0,V1,V4).
-P29(V2,V4):--P30(V0,V1,V2),P31(V0,V1,V4),not-P32(V0,V1,V4).
-P3(V0,V2,V4):-P12(V0,V1),-P20(V2,V3),-P21(V0,V2,V4).
-P4(V0,V1,V3):-P13(V0,V2,V3),P14(V0,V2,V4),P15(V1,V2,V4).
-P5(V1,V3,V4):--P7(V2,V4),P10(V0,V1,V4),-P24(V1,V2,V3).
-P6(V1,V1,V1):-P1(V1),notP8(V1),notP17(V1).
-P7(V2,V4):--P3(V0,V2,V4),not-P22(V0,V2,V4),not-P23(V0,V2,V4).
-P8(V0):--P3(V0,V2,V4),not-P22(V0,V2,V4),not-P23(V0,V2,V4).
-P9(V2,V0,V4):--P3(V0,V2,V4),not-P22(V0,V2,V4),not-P23(V0,V2,V4).
P1(V1):-P2(V1,V3,V4),notP5(V1,V3,V4),not-P16(V1,V3,V4).
P10(V4,V4,V4):--P9(V2,V3,V4),notP18(V2,V3,V4),notP19(V2,V3,V4).
P11(V1,V3,V4):-P8(V1),P9(V2,V3,V4),-P10(V0,V1,V4).
P11(V1,V4,V4):-P3(V0,V2,V4),not-P6(V0,V1,V4),-P10(V0,V1,V4).
P18(V3,V3,V3):-P0(V1,V3),notP1(V1),P8(V1).
P19(V1,V1,V1):--P1(V1),not-P26.
P2(V1,V3,V4):-P12(V0,V1),-P20(V2,V3),-P21(V0,V2,V4).
P27(V1,V2,V3):--P28(V1,V2,V3).
P6(V0,V1,V4):-P0(V1,V3),P3(V0,V2,V4),P11(V1,V3,V4).
<end>
```

**OUTPUT:** 

-P17("Christopher"), -P18("John", "Charles", "Lucas"), -P19("John", "Charles", "Lucas"), -P20("John", "Charles"), -P21("Barbara", "John", "Lucas"), -P24("Christopher", "John", "Charles"), -P28("Christopher", "John", "Charles"), -P30("Barbara", "Christopher", "John"), P12("Barbara", "Christopher"), P13("Barbara", "John", "Charles"), P14("Barbara", "John", "Lucas"), P15("Christopher", "John", "Lucas"), orangeP16("Christopher", "Charles", "Lucas"), P22("Barbara", "John", "Lucas"), P23("Barbara", "John", "Lucas"), P25("Christopher", "Charles"), P26, P31("Barbara", "Christopher", "Lucas"), P32("Barbara", "Christopher", "Lucas"), P2("Christopher", "Charles", "Lucas"), P1("Christopher"), P27("Christopher", "John", "Charles"), -P0("Christopher", "Charles"), -P6("Christopher", "Christopher", "Christopher"), -P3("Barbara", "John", "Lucas"), -P4("Barbara", "Christopher", "Charles"), -P29("John", "Lucas"), -P7("John", "Lucas"), -P8("Barbara"), -P9("John", "Barbara", "Lucas"), P10("Lucas", "Lucas", "Lucas") 

Table 13: The instruction and an example of SymTex-ASC Task. 

25 

Table 14: Experimental results for each task in ONEEVAL. 

|**Dataset**|**AffairQA**|**BioTextQA**|**MatTextQA**|**PharmKGQA**|**ChineseLawFact**|**VersiCode**|
|---|---|---|---|---|---|---|
|**Metric**|**F1 (%)**|**F1 (%)**|**EM (%)**|**F1 (%)**|**ISM@1 (%)**|**F1 (%)**|
|||_Open Source LLMs_|||||
|Qwen2.5-7B|46.00|50.95|37.50|31.55|62.88|17.80|
|Llama3.1-8B|42.00|55.23|55.98|23.53|57.13|11.80|
|GLM4-9B|38.50|80.95|58.10|17.70|66.25|58.50|
|QWQ-32B|45.00|76.67|62.38|45.67|69.00|23.70|
|Llama3.1-70B|40.00|88.57|71.43|34.33|59.38|50.90|
|Qwen2.5-72B|45.00|81.43|62.86|38.09|70.50|35.90|
|Deepseek-V3|42.50|55.71|39.90|39.04|53.87|37.40|
|DeepSeek-R1|45.50|33.81|50.48|31.37|58.00|65.60|
|Llama4-Maverick|43.50|82.38|71.43|40.48|73.12|64.78|
||||_Proprietary LLMs_||||
|GPT-4o|41.00|43.81|61.43|39.23|56.63|66.50|
|GPT-4.1|47.00|46.67|69.05|40.95|73.62|69.72|
|o4-mini|44.50|88.10|71.43|42.86|69.63|67.75|
|Hunyuan-turbo|43.00|85.71|60.95|32.52|83.87|51.70|
|Doubao-pro|40.00|83.33|50.00|27.14|57.50|63.10|
|Grok 3|45.50|80.00|64.29|42.11|54.25|64.00|
|Claude3.7-Sonnet|22.00|78.10|48.80|40.10|60.38|18.20|
|**Dataset**|**PeopleRelQA**|**ReportFixer**|**KCQAD**|**AttributionNLI**|**SymTex-ASC**||
|**Metric**|**F1 (%)**|**F1 (%)**|**EM (%)**|**F1 (%)**|**EM (%)**||
|||_Open Source LLMs_|||||
|Qwen2.5-7B|0.50|17.00|42.00|35.80|24.50||
|Llama3.1-8B|0.20|2.50|50.20|32.60|10.00||
|GLM4-9B|0.20|6.60|55.00|45.10|10.20||
|QWQ-32B|3.00|32.30|65.53|51.20|18.90||
|Llama3.1-70B|2.20|24.20|47.80|54.50|30.70||
|Qwen2.5-72B|2.50|38.90|58.40|57.30|15.30||
|Deepseek-V3|2.60|57.90|56.20|56.00|48.90||
|DeepSeek-R1|6.80|59.70|64.44|71.20|32.20||
|Llama4-Maverick|3.50|32.10|36.20|52.00|30.67||
||||_Proprietary LLMs_||||
|GPT-4o|3.20|44.70|64.00|63.00|23.40||
|GPT-4.1|4.00|53.50|55.60|55.60|22.97||
|o4-mini|3.50|49.30|41.20|41.20|35.95||
|Hunyuan-turbo|1.40|2.20|44.20|47.10|26.90||
|Doubao-pro|0.00|25.30|41.08|60.10|15.20||
|Grok 3|4.70|77.80|48.92|53.60|25.70||
|Claude3.7-Sonnet|0.50|42.30|43.30|62.70|33.80||



## **C Detailed Results on ONEEVAL** 

Table 14 shows the performance of different LLMs on ONEEVAL. Table 15 shows the performance of different LLMs on ONEEVAL-Hard. 

26 

Table 15: Experimental results for each task in ONEEVAL-Hard. 

|**Dataset**|**AffairQA**|**BioTextQA**|**MatTextQA**|**PharmKGQA**|**ChineseLawFact**|**VersiCode**|
|---|---|---|---|---|---|---|
|**Metric**|**Acc (%)**|**F1 (%)**|**EM (%)**|**F1 (%)**|**Acc (%)**|**ISM@1 (%)**|
||||_Open Source LLMs_||||
|Qwen2.5-7B|0.00|26.67|17.79|14.29|21.67|0.00|
|Baichuan2-7B|2.00|26.67|24.29|10.48|18.33|0.00|
|Llama3.1-8B|0.00|28.10|26.79|11.76|31.67|0.00|
|GLM4-9B|0.00|41.43|28.57|9.09|26.67|7.40|
|Baichuan2-13B|0.00|29.05|10.95|7.62|40.00|1.90|
|QWQ-32B|0.00|38.57|28.57|21.63|21.67|1.90|
|Llama3.1-70B|0.00|44.29|35.24|15.42|20.00|4.50|
|Qwen2.5-72B|0.00|40.95|31.90|21.43|6.67|14.00|
|DeepSeek-V3|0.00|28.10|20.10|17.65|6.67|16.50|
|DeepSeek-R1|0.00|9.52|1.43|5.39|21.67|20.40|
|Llama4-Maverick|2.00|40.48|33.81|18.57|16.67|20.00|
||||_Proprietary LLMs_||||
|GPT-4o|0.00|23.33|30.48|19.62|11.67|0.00|
|GPT-4.1|2.00|24.29|35.71|18.57|23.33|17.33|
|o1|0.00|42.45|33.65|21.15|18.33|8.80|
|o3|2.00|41.90|38.46|22.49|23.33|29.58|
|o4-mini|0.00|43.33|36.67|21.90|21.67|18.09|
|Hunyuan-turbo|4.00|41.43|29.05|15.53|23.33|9.80|
|Doubao-pro|0.00|42.38|22.86|11.90|16.67|11.10|
|Claude3.7-Sonnet|0.00|40.00|22.01|18.36|23.33|3.70|
|Grok 3|3.00|40.00|29.52|19.62|11.67|14.80|
|Gemini-2.5-pro|2.00|41.90|30.48|21.53|26.67|29.92|
|**Dataset**|**PeopleRelQA**|**ReportFixer**|**KCQAD**|**AttributionNLI**|**SymTex-ASC**||
|**Metric**|**F1 (%)**|**F1 (%)**|**EM (%)**|**F1 (%)**|**EM (%)**||
||||_Open Source LLMs_||||
|Qwen2.5-7B|2.00|0.00|2.20|10.40|22.10||
|Baichuan2-7B|0.00|0.00|4.81|0.00|0.50||
|Llama3.1-8B|0.00|0.00|4.20|2.10|9.00||
|GLM4-9B|0.00|3.10|5.00|2.10|22.10||
|Baichuan2-13B|0.00|0.00|3.40|0.00|1.00||
|QWQ-32B|0.40|12.00|11.62|16.70|17.30||
|Llama3.1-70B|0.80|9.80|4.40|8.30|32.20||
|Qwen2.5-72B|8.10|21.30|5.40|12.50|15.00||
|DeepSeek-V3|2.00|37.80|5.00|12.50|49.30||
|DeepSeek-R1|8.00|55.60|8.89|25.00|32.70||
|Llama4-Maverick|6.00|13.30|39.32|12.50|28.43||
||||_Proprietary LLMs_||||
|GPT-4o|2.50|39.30|11.00|18.80|22.40||
|GPT-4.1|4.00|47.30|51.46|25.00|22.28||
|o1|8.00|36.30|22.60|27.10|26.20||
|o3|20.00|67.70|56.82|22.91|28.55||
|o4-mini|4.00|74.20|42.54|20.83|39.76||
|Hunyuan-turbo|3.00|0.80|3.40|2.10|24.90||
|Doubao-pro|0.00|12.20|2.61|6.30|16.70||
|Claude3.7-Sonnet|0.40|3.20|3.60|18.80|33.80||
|Grok 3|6.00|70.80|4.33|14.60|24.80||
|Gemini-2.5-pro|8.00|20.00|22.33|20.83|27.37||



27 

