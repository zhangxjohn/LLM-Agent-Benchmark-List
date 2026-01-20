# LLM-Agent-Benchmark-List

🤗**We greatly appreciate any contributions via PRs, issues, emails, or other methods.**

⏳ **Continuous update...**

## :book: Introduction

In the swiftly evolving landscape of artificial intelligence, Large Language Models (LLMs) have emerged as a pivotal cornerstone, revolutionizing how we interact with and harness the power of natural language processing.  However, as LLMs gain widespread application in both research and industry sectors, the imperative shifts towards evaluating their efficacy rather than perpetuating a cycle of unbridled performance iterations. This paradigm shift raises critical questions: *i) what to evaluate?  ii) where to evaluate? iii)How to evaluate?* Diverse research endeavors have proposed varying interpretations and methodologies in response to these queries. The aim of this work is to methodically review and organize benchmarks that are both LLMs and agent-powered, thereby providing a streamlined resource for those journeying towards Artificial General Intelligence (AGI).



## :dizzy: List

#### Survey

- [2023/07] **A Survey on Evaluation of Large Language Models.** *Yupeng Chang ( Jilin University) et al. arXiv.* [[paper](https://arxiv.org/pdf/2307.03109.pdf)] [[project page](https://github.com/MLGroupJLU/LLM-eval-survey)]
- [2023/10] **Evaluating Large Language Models: A Comprehensive Survey.** *Zishan Guo (Tianjin) et al. arXiv.* [[paper](https://arxiv.org/pdf/2310.19736.pdf)] [[project page](https://github.com/tjunlp-lab/Awesome-LLMs-Evaluation-Papers)]

----------------------

#### ToolUse

- [2023/10] **API-Bank: A Comprehensive Benchmark for Tool-Augmented LLMs.** *Minghao Li (Alibaba) et al. arXiv.* [[paper](https://arxiv.org/pdf/2304.08244.pdf)] [[project page](https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/api-bank)]
- [2023/05] **On the Tool Manipulation Capability of Open-source Large Language Models.** *Qiantong Xu (SambaNova) et al. arXiv.* [[paper](https://arxiv.org/pdf/2305.16504.pdf)] [[project page](https://github.com/sambanova/toolbench)]
- [2023/10] **ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs.**  *Yujia Qin (Tsinghua) et al. arXiv.* [[paper](https://arxiv.org/pdf/2307.16789.pdf)] [[project page](https://github.com/OpenBMB/ToolBench)]
- [2024/01] **T-Eval: Evaluating the Tool Utilization Capability of Large Language Models Step by Step.** *Zehui Chen (USTC) et al. arXiv.* [[paper](https://arxiv.org/pdf/2312.14033.pdf)] [[project page](https://github.com/open-compass/T-Eval)]

----------------------------

#### Reasoning

- [2024/01] **NPHardEval: Dynamic Benchmark on Reasoning Ability of Large Language Models via Complexity Classes.** *Lizhou Fan (Michigan) et al. arXiv.* [[paper](https://arxiv.org/pdf/2312.14890v2.pdf)] [[project page](https://github.com/casmlab/NPHardEval)]
- [2024/01] **A Survey of Reasoning with Foundation Models.** *Jiankai Sun (CUHK) et al. arXiv.* [[paper](https://arxiv.org/pdf/2312.11562.pdf)] [[project page](https://github.com/reasoning-survey/Awesome-Reasoning-Foundation-Models)]
- [2024/01] **LLMs for Relational Reasoning: How Far are We?** *Zhiming Li (NTU) et al. arXiv.* [[paper](https://arxiv.org/pdf/2401.09042.pdf)]
- [2023/08] **Are Large Language Models Really Good Logical Reasoners? A Comprehensive Evaluation and Beyond.** *Fangzhi Xu ( Xi’an Jiaotong University) et al. arXiv.* [[paper](https://arxiv.org/pdf/2306.09841.pdf)] [[project page](https://github.com/DeepReasoning/NeuLR)]
- [2023/11] **PlanBench: An Extensible Benchmark for Evaluating Large Language Models on Planning and Reasoning about Change.** *Karthik Valmeekam (ASU) et al. arXiv.* [[paper](https://arxiv.org/pdf/2206.10498.pdf)] [[project page](https://github.com/karthikv792/LLMs-Planning)]
- [2023/12] **Understanding Social Reasoning in Language Models with Language Models.** *Kanishk Gandhi (Stanford) et al. arXiv.* [[paper](https://arxiv.org/pdf/2306.15448.pdf)] [[project page](https://sites.google.com/view/social-reasoning-lms)]

------------------------------------

#### Knowledge

- [2023/12] **Trends in Integration of Knowledge and Large Language Models: A Survey and Taxonomy of Methods, Benchmarks, and Applications.** *Zhangyin Feng (Harbin Institute of Technology) et al. arXiv.* [[paper](https://arxiv.org/pdf/2311.05876.pdf)]
- [2023/09] **Benchmarking Large Language Models in Retrieval-Augmented Generation.** *Jiawei Chen (CIPL) et al. arXiv.* [[paper](https://arxiv.org/pdf/2309.01431.pdf)]

-----------------------------------

#### Graph

- [2023/09] **Evaluating Large Language Models on Graphs: Performance Insights and Comparative Analysis.** *Chang Liu (Colorado School of Mines) et al. arXiv.* [[paper](https://arxiv.org/pdf/2308.11224.pdf)] [[project page](https://github.com/Ayame1006/LLMtoGraph)]

-----------------------------------------

#### Video

- [2023/10] **EvalCrafter: Benchmarking and Evaluating Large Video Generation Models.** *Yaofang Liu (Tencent) et al. arXiv.* [[paper](https://arxiv.org/pdf/2310.11440.pdf)] [[project page](https://evalcrafter.github.io/)]

-----------------------------------

#### Code

- [2023/08] **Evaluating Instruction-Tuned Large Language Models on Code Comprehension and Generation.** *Zhiqiang Yuan (Fudan) et al. arXiv.* [[paper](https://arxiv.org/pdf/2308.01240.pdf)]
- [2024/01] **CRUXEval: A Benchmark for Code Reasoning, Understanding and Execution.** *Alex Gu (Meta) et al. arXiv.* [[paper](https://arxiv.org/pdf/2401.03065.pdf)]
- [2023/10] **Is Your Code Generated by ChatGPT Really Correct? Rigorous Evaluation of Large Language Models for Code Generation.** *Jiawei Liu (UIUC) et al. arXiv.* [[paper](https://arxiv.org/pdf/2305.01210.pdf)] [[project page](https://github.com/evalplus/evalplus)]
- [2023/10] **SWE-bench: Can Language Models Resolve Real-World GitHub Issues?** *Carlos E. Jimenez (Princeton University) et al. ICLR.* [[paper](https://arxiv.org/abs/2310.06770)] [[project page](https://github.com/princeton-nlp/SWE-bench)]
- [2024/06] **BigCodeBench: Benchmarking Code Generation with Diverse Function Calls and Complex Instructions.** *Terry Yue Zhuo (Monash University) et al. arXiv.* [[paper](https://arxiv.org/pdf/2406.15877)] [[project page](https://github.com/bigcode-project/bigcodebench)]
- [2025/01] **How Should We Build A Benchmark? Revisiting 274 Code-Related Benchmarks For LLMs.** *Jialun Cao (The Hong Kong University of Science and Technology) et al. arXiv.* [[paper](https://arxiv.org/pdf/2501.10711)]

---------------------------------

#### Alignment

- [2023/12] **AlignBench: Benchmarking Chinese Alignment of Large Language Models.** *Xiao Liu (KEG) et al. arXiv.* * [[paper](https://arxiv.org/pdf/2311.18743.pdf)] [[project page](https://github.com/THUDM/AlignBench)]

----------------------------

#### Agent
- [2025/08] **OdysseyBench: Evaluating LLM Agents on Long-Horizon Complex Office Application Workflows** *Weixuan Wang (University of Edinburgh, Microsoft) et al. arXiv.* [[paper](https://arxiv.org/abs/2508.09124)]
- [2024/02] **Agent-Pro: Learning to Evolve via Policy-Level Reflection and Optimization** *Wenqi Zhang (Zhejiang University) et al. arXiv.* [[paper](https://arxiv.org/abs/2402.17574)] [[project page](https://github.com/zwq2018/Agent-Pro)]
- [2024/01] **Self-Contrast: Better Reflection Through Inconsistent Solving Perspectives** *Wenqi Zhang (Zhejiang University) et al. arXiv.* [[paper](https://arxiv.org/abs/2401.02009)]
- [2023/10] **AgentBench: Evaluating LLMs as Agents.**  *Xiao Liu (Tsinghua) et al. arXiv.* [[paper](https://arxiv.org/abs/2308.03688)] [[project page](https://github.com/THUDM/AgentBench)]
- [2023/08] **AgentSims: An Open-Source Sandbox for Large Language Model Evaluation.** *Jiaju Lin (PTA Studio) et al. arXiv.* [[paper](https://arxiv.org/pdf/2308.04026.pdf)] 
- [2023/12] **Large Language Models Empowered Agent-based Modeling and Simulation: A Survey and Perspectives.** *Chen Gao (Tsinghua) et al. arXiv.* * [[paper](https://arxiv.org/pdf/2312.11970.pdf)]
- [2024/01] ***CharacterEval*: A Chinese Benchmark for Role-Playing Conversational Agent Evaluation.** *Quan Tu (GSAI) et al. arXiv.* [[paper](https://arxiv.org/pdf/2401.01275.pdf)] [[project page](https://github.com/morecry/CharacterEval)]
- [2023/10] **WebArena: A Realistic Web Environment for Building Autonomous Agents.** *Shuyan Zhou (CMU) et al. arXiv.* [[paper](https://arxiv.org/pdf/2307.13854.pdf)] [[project page](https://webarena.dev/)]
- [2024/01] **Evaluating Language-Model Agents on Realistic Autonomous Tasks.** *Megan Kinniment (METR ) et al. arXiv.* [[paper](https://arxiv.org/pdf/2312.11671.pdf)] [[project page](https://metr.org/)]
- [2023/11] **MAgIC: Investigation of Large Language Model Powered Multi-Agent in Cognition, Adaptability, Rationality and Collaboration.** *Lin Xu (NUS) et al. arXiv.* [[paper](https://arxiv.org/pdf/2311.08562.pdf)] [[project page](https://github.com/cathyxl/MAgIC)]
- [2023/12] **LLF-Bench: Benchmark for Interactive Learning from Language Feedback.** *Ching-An Cheng (Microsoft) et al. arXiv.* [[paper](https://arxiv.org/pdf/2312.06853.pdf)] [[project page](https://microsoft.github.io/LLF-Bench/)]
- [2024/01] **AgentBoard: An Analytical Evaluation Board of Multi-turn LLM Agents.** *Chang Ma (The University of Hong Kong) et al. arXiv.* [[paper](https://arxiv.org/pdf/2401.13178)] [[project page](https://github.com/hkust-nlp/AgentBoard)]
- [2024/07] **AppWorld: A Controllable World of Apps and People for Benchmarking Interactive Coding Agents*** *Harsh Trivedi (Stony Brook University) et al. ACL.* [[paper](https://arxiv.org/pdf/2407.18901)][[project page](https://appworld.dev/)]
- [2024/03] **OSWORLD: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments** *Tianbao Xie (The University of Hong Kong) et al. arXiv.* [[paper](https://arxiv.org/abs/2404.07972)][[project page](https://os-world.github.io/)]
- [2025/03] **Benchmarking Agentic Workflow Generation.** *Shuofei Qiao (Zhejiang University) et al. ICLR.* [[paper](https://arxiv.org/abs/2410.07869)] [[project page](https://github.com/zjunlp/WorfBench)]
- [2025/05] **Terminal-Bench: A Benchmark for AI Agents in Terminal Environments.** *Mike Merill (Standford) et al.* [[project page](https://github.com/laude-institute/terminal-bench)]
- [2025/05] **Beyond Static Testbeds: An Interaction-Centric Agent Simulation Platform for Dynamic Recommender Systems** *Song Jin (RUC) et al. EMNLP.* [[paper](https://aclanthology.org/2025.emnlp-main.956/)][[project page](https://github.com/jinsong8/RecInter)]
- [2025/06] **CitySim: Modeling Urban Behaviors and City Dynamics with Large-Scale LLM-Driven Agent Simulation** *Nicolas Bougie (Woven by Toyota) et al. arXiv.* [[paper](https://arxiv.org/abs/2506.21805)]
- [2025/08] **VimGolf-Gym: OpenAI Gym Style VimGolf Environment and Benchmark** *James Brown (Cybergod AGI Research)* [[project page](https://github.com/james4Ever0/vimgolf-gym)]

-----------------------------------

#### Multimodal

- [2023/11] **M3Exam: A Multilingual, Multimodal, Multilevel Benchmark for Examining Large Language Models.** *Wenxuan Zhang (Alibaba) et al. arXiv.* [[paper](https://arxiv.org/pdf/2306.05179.pdf)] [[project page](https://github.com/DAMO-NLP-SG/M3Exam?tab=readme-ov-file)]
- [2023/11] **A Multitask, Multilingual, Multimodal Evaluation of ChatGPT on Reasoning, Hallucination, and Interactivity.** *Yejin Bang (CAiRE) et al. arXiv.* [[paper](https://arxiv.org/pdf/2302.04023.pdf)] [[project page](https://github.com/HLTCHKUST/chatgpt-evaluation)]
- [2023/12] **MME: A Comprehensive Evaluation Benchmark for Multimodal Large Language Models.** *Chaoyou Fu (Tencent) et al. arXiv.* [[paper](https://arxiv.org/pdf/2306.13394.pdf)] [[project page](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation)]
- [2024/01] **Q-Bench: A Benchmark for General-Purpose Foundation Models on Low-level Vision.** *Haoning Wu (Nanyang Technological University) et al. arXiv.* [[paper](https://arxiv.org/abs/2309.14181)] [[project page](https://github.com/Q-Future/Q-Bench?tab=readme-ov-file)]
- [2023/10] **Beyond Task Performance: Evaluating and Reducing the Flaws of Large Multimodal Models with In-Context Learning.** *Mustafa Shukor (Sorbonne University) et al. ICLR.* [[paper](https://arxiv.org/pdf/2310.00647.pdf)] [[project page](https://github.com/mshukor/EvALign-ICL)]
- [2023/12] **InfiMM-Eval: Complex Open-ended Reasoning Evaluation for Multi-modal Large Language Models.** *Xiaotian Han (ByteDance) et al. arXiv.* * [[paper](https://arxiv.org/pdf/2311.11567.pdf)] [[project page](https://infimm.github.io/InfiMM-Eval/)]

----------------------------------

#### Comprehensive Evaluation

- [2024/06] **LiveBench: A Challenging, Contamination-Free LLM Benchmark.** *Colin White (Abacus.AI) et al. arXiv.* [[paper](https://arxiv.org/pdf/2406.19314)] [[project page](https://livebench.ai/)]
- [2024/10] **JudgeBench: A Benchmark for Evaluating LLM-based Judges.** *Sijun Tan (UC Berkeley) et al. arXiv.* [[paper](https://arxiv.org/pdf/2410.12784)] [[project page](https://github.com/ScalerLab/JudgeBench)]
- [2024/06] **MixEval: Deriving Wisdom of the Crowd from LLM Benchmark Mixtures.** *Jinjie Ni (National University of Singapore) et al. NeurIPS.* [[paper](https://arxiv.org/pdf/2406.06565)] [[project page](https://mixeval.github.io/)]
- [2025/02] **LLM-Powered Benchmark Factory: Reliable, Generic, and Efficient.** *Peiwen Yuan (Beijing Institute of Technology) et al. arXiv.* [[paper](https://arxiv.org/pdf/2502.01683)][[project page](https://github.com/ypw0102/BenchMaker)]
- [2024/10] **DetectRL: Benchmarking LLM-Generated Text Detection in Real-World Scenarios.** *Junchao Wu (University of Macau) et al. arXiv.* [[paper](https://arxiv.org/abs/2410.23746)]
- [2025/02] **Auto-Bench: An Automated Benchmark for Scientific Discovery in LLMs.** *Tingting Chen (National University of Singapore) et al. arXiv.* [[paper](https://arxiv.org/pdf/2502.15224)]

----------------------

#### Others

- [2023/11] **Benchmarking Foundation Models with Language-Model-as-an-Examiner.** *Yushi Bai (Tsinghua) et al. arXiv.* [[paper](https://arxiv.org/pdf/2306.04181.pdf)]
- [2023/08] **LLMeBench: A Flexible Framework for Accelerating LLMs Benchmarking.** *Fahim Dalvi (HBKU) et al. arXiv.* [[paper](https://arxiv.org/pdf/2308.04945.pdf)] [[project page](https://github.com/qcri/LLMeBench/)]
- [2024/01] **PromptBench: A Unified Library for Evaluation of Large Language Models.** *Kaijie Zhu (MSRA) et al. arXiv.* [[paper](https://arxiv.org/pdf/2312.07910.pdf)] [[project page](https://github.com/microsoft/promptbench)]
- [2023/11] **TencentLLMEval: A Hierarchical Evaluation of Real-World Capabilities for Human-Aligned LLMs.** *Shuyi Xie (Tencent) et al. arXiv.* [[paper](https://arxiv.org/pdf/2311.05374.pdf)] [[project page](https://github.com/xsysigma/TencentLLMEval)]
- [2024/01] **Benchmarking LLMs via Uncertainty Quantification.** *Fanghua Ye (Tencent) et al. arXiv.* [[paper](https://arxiv.org/pdf/2401.12794.pdf)] [[project page](https://github.com/smartyfh/LLM-Uncertainty-Bench)]
- [2023/12] **LLMEval: A Preliminary Study on How to Evaluate Large Language Models.** *Yue Zhang (Fudan) et al. arXiv.* [[paper](https://arxiv.org/pdf/2312.07398.pdf)] [[project page](https://github.com/llmeval)]
- [2023/08] **Large Language Models are not Fair Evaluators.** *Peiyi Wang (Peking) et al. arXiv.* [[paper](https://arxiv.org/pdf/2305.17926.pdf)] [[project page](https://github.com/i-Eval/FairEval)]
- [2023/09] **Evaluating Cognitive Maps and Planning in Large Language Models with CogEval.** *Ida Momennejad (Microsoft)  et al. arXiv.*  [[paper](https://arxiv.org/pdf/2309.15129.pdf)]
- [2023/04] **SocialDial: A Benchmark for Socially-Aware Dialogue Systems.** *Haolan Zhan (Monash) et al. SIGIR.* [[paper](https://arxiv.org/pdf/2304.12026.pdf)] [[project page](https://github.com/zhanhl316/SocialDial)]
- [2023/07] **Emotional Intelligence of Large Language Models.** *Xuena Wang (Tsinghua) et al. arXiv.* [[paper](https://arxiv.org/ftp/arxiv/papers/2307/2307.09042.pdf)]
- [2024/01] **Assessing and Understanding Creativity in Large Language Models.** *Yunpu Zhao (USTC) et al. arXiv.* [[paper](https://arxiv.org/pdf/2401.12491.pdf)]
- [2023/11] **Rethinking Benchmark and Contamination for Language Models with Rephrased Samples.** *Shuo Yang (UC Berkeley) et al. arXiv.* [[paper](https://arxiv.org/pdf/2311.04850.pdf)] [[project page](https://github.com/lm-sys/llm-decontaminator)]
- [2023/06] **PandaLM: An Automatic Evaluation Benchmark for LLM Instruction Tuning Optimization.** *Yidong Wang (Peking University) et al. arXiv.* [[paper](https://arxiv.org/pdf/2306.05087)] [[project page](https://github.com/WeOpenML/PandaLM)]
- [2024/09] **Do These LLM Benchmarks Agree? Fixing Benchmark Evaluation with BenchBench.** *Yotam Perlitz (IBM Research AI) et al. arXiv.* [[paper](https://arxiv.org/pdf/2407.13696)] [[project page](https://github.com/IBM/benchbench)]
- [2024/03] **Table Meets LLM: Can Large Language Models Understand Structured Table Data? A Benchmark and Empirical Study.** *Yuan Sui (National University of Singapore) et al. WSDM.* [[paper](https://dl.acm.org/doi/pdf/10.1145/3616855.3635752?trk=public_post_comment-text)]

----------------------

#### Dataset

- [2023/10] **Investigating Table-to-Text Generation Capabilities of LLMs in Real-World Information Seeking Scenarios.** *Yilun Zhao (Yale) EMNLP.* [[paper](https://aclanthology.org/2023.emnlp-industry.17.pdf)] [[project page](https://github.com/yale-nlp/LLM-T2T)]
- [2023/10] **Empower Large Language Model to Perform Better on Industrial Domain-Specific Question Answering.** *Fangkai Yang (Microsoft) EMNLP.*[[paper](https://arxiv.org/pdf/2305.11541.pdf)] [[project page](https://github.com/microsoft/Microsoft-Q-A-MSQA-)]

------------------------------

#### ZhiHu

- [2023/11] **LLM Evaluation 如何评估一个大模型？** *Isaac 张雯轩* [[project page](https://zhuanlan.zhihu.com/p/644373658)]
- [2023/11] **基于LLM Agent评估研究** *龚旭东* [[project page](https://zhuanlan.zhihu.com/p/669325175)]



