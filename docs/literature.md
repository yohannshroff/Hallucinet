# Literature for the final report

Fourteen papers from 2023 or later. Identifiers, titles, authors and venues were
checked against live search results on 2026-10-19, and the full text of each
paper was then read to fill in the entries below (the numbers are copied from the
paper's own tables or text; the table or section is given where it was noted).
This is a working draft: **each team member must still read the papers they
cite** and rewrite the entries in their own words for the report.

`preprint` means the version cited is an arXiv preprint (the paper may also
appear at the venue named).

## Entries (one per row of report Table 2.1)

### 1. Gao et al., 2023 — arXiv:2312.10997 (preprint)
**Retrieval-Augmented Generation for Large Language Models: A Survey**
- **Problem:** LLMs hallucinate, have outdated knowledge and untraceable reasoning; RAG adds external knowledge.
- **Method:** Survey. Sorts RAG into Naive, Advanced and Modular RAG and reviews retrieval, generation and augmentation techniques.
- **Data / metrics:** Review of existing benchmarks and evaluation frameworks (no new experiments).
- **Key findings:** Descriptive; no headline numbers.
- **Limitations:** Lists open challenges (retrieval quality, generation faithfulness, augmentation) rather than testing them.
- **For us:** Our vector RAG is the "Naive RAG" baseline; vocabulary for the design chapter.

### 2. Peng et al., 2024 — arXiv:2408.08921 (preprint)
**Graph Retrieval-Augmented Generation: A Survey**
- **Problem:** Plain text RAG ignores relationships between entities.
- **Method:** Survey. Formalises GraphRAG as graph-based indexing, graph-guided retrieval and graph-enhanced generation.
- **Data / metrics:** Reviews downstream tasks, applications and evaluation methods (no new experiments).
- **Key findings:** Descriptive; no headline numbers.
- **Limitations:** Field described as early stage; evaluation practice varies.
- **For us:** Frames our hybrid design (graph index, graph search, merged context).

### 3. Edge et al., 2024 — arXiv:2404.16130 (preprint)
**From Local to Global: A Graph RAG Approach to Query-Focused Summarization**
- **Problem:** Vector RAG cannot answer questions about a whole corpus ("global sensemaking").
- **Method:** LLM builds an entity graph, groups it into communities, pre-writes community summaries, and combines partial answers at query time.
- **Data / metrics:** Two corpora of about 1 million tokens each (podcast transcripts, news articles); LLM-judged head-to-head win rates for comprehensiveness and diversity.
- **Key findings:** Versus vector RAG, global GraphRAG won comprehensiveness 72–83% (podcasts) and 72–80% (news), p<.001, and diversity 75–82% and 62–71%.
- **Limitations (stated):** Only two corpora of this size and type; fabrication rates not compared (they suggest SelfCheckGPT-style checks). The judge is an LLM.
- **For us:** Targets whole-corpus summaries; our questions are fact lookups, so the benefit may not carry over. Also an LLM-judged evaluation, which we avoid.

### 4. Jiménez Gutiérrez et al., 2024 — arXiv:2405.14831 (NeurIPS 2024)
**HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models**
- **Problem:** Standard RAG struggles to combine facts spread across passages (multi-hop).
- **Method:** LLM extracts a knowledge graph from the passages; Personalized PageRank over it ranks passages for a query.
- **Data / metrics:** MuSiQue, 2WikiMultiHopQA, HotpotQA (1,000-question dev sets); Recall@2/@5, exact match, F1.
- **Key findings (Table 2):** Single-step Recall@5 on 2WikiMultiHopQA 89.5 (HippoRAG with Contriever) against 68.2 for ColBERTv2; MuSiQue 52.1 vs 49.2; HotpotQA comparable (76.2 vs 79.3, i.e. slightly lower). Abstract claims gains of up to 20%.
- **Limitations (stated):** Most errors come from entity and relation extraction; off-the-shelf components; scalability not yet validated.
- **For us:** The closest "graph helps multi-hop" claim. Our graph is much weaker (mostly co-mention edges, built with a 4B model), and the paper itself says extraction quality is the main error source. We report our multi-hop results with that in mind.

### 5. Asai et al., 2024 — arXiv:2310.11511 (ICLR 2024)
**Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection**
- **Problem:** Always retrieving a fixed number of passages can add noise, and outputs need not follow the passages.
- **Method:** Train one LM to decide when to retrieve and to emit reflection tokens that critique relevance, support and quality.
- **Data / metrics:** PopQA, TriviaQA, PubHealth, ARC-Challenge, biography generation (FactScore), ASQA (citation precision and recall).
- **Key findings (Table 2):** Self-RAG 7B: PopQA 54.9, PubHealth 72.4, ARC 67.3, biography FactScore 81.2, citation precision 66.9 and recall 67.8; ChatGPT scores PopQA 29.3 and PubHealth 70.1.
- **Limitations:** Requires training the LM with special tokens; not stated as a separate section in the text we read.
- **For us:** Contrast with our fixed pipeline and external NLI check; its citation precision/recall is the metric idea behind our citation accuracy.

### 6. Yan et al., 2024 — arXiv:2401.15884 (preprint)
**Corrective Retrieval Augmented Generation**
- **Problem:** RAG fails when retrieval returns the wrong documents.
- **Method:** A lightweight evaluator scores retrieved documents and triggers one of three actions (use, correct with web search, or both); a decompose-then-recompose step keeps key sentences.
- **Data / metrics:** PopQA (accuracy), biography generation (FactScore), PubHealth, Arc-Challenge.
- **Key findings:** With a Self-RAG LLaMA2-7B base, CRAG beat plain RAG by 7.0% accuracy on PopQA, 14.9% FactScore on biographies, 36.6% on PubHealth and 15.4% on Arc-Challenge.
- **Limitations (stated):** Needs a fine-tuned external retrieval evaluator.
- **For us:** Retrieval failure is the main loss in our results (vector search misses the gold pages for about 30% of questions at k=5). Corrective retrieval is the natural next step.

### 7. Chen et al., 2024 — arXiv:2309.01431 (AAAI 2024; DOI 10.1609/AAAI.V38I16.29728)
**Benchmarking Large Language Models in Retrieval-Augmented Generation**
- **Problem:** No rigorous evaluation of how RAG affects different LLMs.
- **Method:** RGB benchmark (English and Chinese) with four test sets: noise robustness, negative rejection, information integration, counterfactual robustness.
- **Data / metrics:** Six LLMs; accuracy, rejection rate, error detection.
- **Key findings:** Negative rejection is weak: the best rejection rates were 45% (English) and 43.33% (Chinese) when only noise documents were given (Table 3). Models also struggle to integrate information and to resist false documents.
- **Limitations:** Models often do not follow the rejection instruction exactly, which makes rejection hard to detect automatically.
- **For us:** "Negative rejection" is our unanswerable category. Our 4B model's refusal behaviour (graph-only and vector-only 100% on our unanswerable questions) can be compared with their weak numbers; ours is easier because the book either has the answer or does not, and our context is small.

### 8. Es et al., 2024 — arXiv:2309.15217 (EACL 2024 demonstrations)
**RAGAs: Automated Evaluation of Retrieval Augmented Generation**
- **Problem:** Evaluating RAG normally needs reference answers.
- **Method:** Reference-free metrics for faithfulness, answer relevance and context relevance, computed with an LLM.
- **Data / metrics:** The WikiEval set they built; agreement with human annotators in pairwise comparisons.
- **Key findings (Table 1):** Agreement with humans: faithfulness 0.95, answer relevance 0.78, context relevance 0.70, ahead of GPT Score and GPT Ranking baselines.
- **Limitations:** Small, Wikipedia-based evaluation set; relies on an LLM judge.
- **For us:** Our trust score is a faithfulness metric of the same kind, but built on an NLI model rather than an LLM. They validate theirs against humans, which is what the review asks of us.

### 9. Saad-Falcon et al., 2024 — arXiv:2311.09476 (NAACL 2024)
**ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems**
- **Problem:** Hand-annotating RAG outputs is costly.
- **Method:** Train small LM judges on synthetic data; use a small human-labelled set with prediction-powered inference to give confidence intervals.
- **Data / metrics:** Eight tasks (KILT, SuperGLUE, AIS); context relevance, answer faithfulness, answer relevance.
- **Key findings:** Accurate with only a few hundred human annotations; they report needing 78% fewer annotations than the baseline approach.
- **Limitations (stated):** Still needs roughly 150–300 human-labelled points, and a domain-aware annotator.
- **For us:** Supports using a small human-labelled validation sample (our 50–60 items with two annotators) to check an automatic scorer.

### 10. Min et al., 2023 — arXiv:2305.14251 (EMNLP 2023)
**FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation**
- **Problem:** Long answers mix true and false statements, so one right/wrong label is too coarse.
- **Method:** Split text into atomic facts; score the share supported by a knowledge source; an automatic estimator uses retrieval plus a strong LM.
- **Data / metrics:** People biographies against Wikipedia; human agreement and error rate of the estimator.
- **Key findings:** ChatGPT scores 58.3% (InstructGPT 42.5%, PerplexityAI 71.5%); the automatic estimator has under 2% error against humans.
- **Limitations (stated):** Only biographies and Wikipedia; the estimator is not perfect on individual judgements.
- **For us:** Our claim-level hallucination rate follows this idea but splits at sentence level (coarser) and checks against the gold book pages.

### 11. Manakul, Liusie, Gales, 2023 — arXiv:2303.08896 (EMNLP 2023; ACL 2023.emnlp-main.557)
**SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models**
- **Problem:** Detect hallucinations without model probabilities or an external database.
- **Method:** Sample several answers; inconsistent statements are likely hallucinated (variants: BERTScore, QA, n-gram, NLI, prompting).
- **Data / metrics:** 238 GPT-3 biography passages from WikiBio; sentence-level AUC-PR and passage-level correlation.
- **Key findings (Table 2):** Non-factual sentence AUC-PR 92.50 with NLI and 93.42 with prompting, against a random baseline of 72.96; passage-level Pearson 74.14 (NLI) and 78.32 (prompting).
- **Limitations (stated):** Mostly biographical text; sentence-level labels; the best variant is expensive.
- **For us:** A cheap extra signal we could add: our three runs per question already give several samples per question.

### 12. Li et al., 2023 — arXiv:2305.11747 (EMNLP 2023; ACL 2023.emnlp-main.397)
**HaluEval: A Large-Scale Hallucination Evaluation Benchmark for Large Language Models**
- **Problem:** How often do LLMs hallucinate, and can they recognise it?
- **Method:** Sample-then-filter generation of hallucinated examples with ChatGPT, plus human annotation of ChatGPT responses.
- **Data / metrics:** QA, dialogue, summarisation and general queries; classification accuracy.
- **Key findings:** About 19.5% of ChatGPT responses contained unverifiable fabricated content; ChatGPT identified hallucinations in QA with only 62.59% accuracy, rising to 76.83% when given external knowledge.
- **Limitations (stated):** Hallucinated samples are limited by ChatGPT's own ability to follow the generation instructions; the benchmark tests recognition rather than avoidance.
- **For us:** Shows that giving the model evidence helps, which is the premise of our RAG baseline, and that a model's own judgement of hallucination is unreliable (why we use NLI plus human validation).

### 13. Huang et al., 2024/2025 — arXiv:2311.05232 (ACM TOIS)
**A Survey on Hallucination in Large Language Models: Principles, Taxonomy, Challenges, and Open Questions**
- **Problem:** A clear taxonomy of LLM hallucination, its causes, detection and mitigation.
- **Method:** Survey. Splits hallucination into factuality hallucination (conflicts with real-world facts) and faithfulness hallucination (departs from the user's input or the given context); notes the older intrinsic/extrinsic split from task-specific models.
- **Data / metrics:** Reviews detection methods and benchmarks; no new experiments.
- **Key findings:** Descriptive; includes limits of retrieval-augmented LLMs in preventing hallucination.
- **Limitations:** Survey only.
- **For us:** Source for our operational definition: our claim-level rate measures faithfulness to the retrieved context (trust score) and factuality against the gold book pages.

### 14. Liu et al., 2024 — arXiv:2307.03172 (TACL)
**Lost in the Middle: How Language Models Use Long Contexts**
- **Problem:** Do models actually use everything in a long context?
- **Method:** Move the relevant document to different positions in multi-document QA and key-value retrieval tasks.
- **Data / metrics:** Multi-document QA accuracy across positions and context lengths; closed-book and oracle baselines (Table 1: GPT-3.5-Turbo closed-book 56.1%, oracle 88.3%).
- **Key findings:** Accuracy is best when the answer is at the start or end and falls in the middle; GPT-3.5-Turbo can drop by more than 20%, and in the 20- and 30-document settings it falls below its closed-book 56.1%.
- **Limitations:** Not stated as a separate section in the text we read; tested on specific tasks and models.
- **For us:** Why top-k matters for a small 4B model: more passages are not always better. Our hybrid context (about 19 graph facts plus passages) is long for this reason.

## Gap statement (Section 2.3)

Graph-based RAG is usually evaluated on large general corpora with automatically
built graphs, and rarely with questions split by type or with the "answer not in
the corpus" case measured. We test vector, graph and hybrid retrieval against a
closed-book baseline on a single history book, on questions written independently
of the retrieval system, with unanswerable and multi-hop questions reported
separately, and with a human check of the automatic hallucination scorer.

## Still to do by hand

- Each member reads the papers they cite and rewrites the entries in their own words.
- The review asks for a search strategy (databases, keywords, dates, counts).
  Record what you actually searched; the list above is only what was verified.
