# Literature for the final report

Fourteen papers from 2023 or later. Each identifier, title, author list, venue and
date was checked against a live web search (arXiv / ACL Anthology / publisher
page) on 2026-10-19. **The team must still open and read each paper before
citing it** (the review treats unread references as a risk): the "what it gives
us" column is a short note on relevance, not a substitute for reading.

| # | Reference (IEEE style) | Venue / ID | Theme | What it gives us |
|---|---|---|---|---|
| 1 | Y. Gao *et al.*, "Retrieval-Augmented Generation for Large Language Models: A Survey," 2023. | arXiv:2312.10997 (preprint) | RAG overview | Naive / advanced / modular RAG vocabulary; our vector RAG is the "naive" baseline |
| 2 | B. Peng *et al.*, "Graph Retrieval-Augmented Generation: A Survey," 2024. | arXiv:2408.08921 (preprint) | GraphRAG | Three stages (graph indexing, graph-guided retrieval, graph-enhanced generation); frames our hybrid design |
| 3 | D. Edge *et al.*, "From Local to Global: A Graph RAG Approach to Query-Focused Summarization," 2024. | arXiv:2404.16130 (preprint) | GraphRAG | LLM-built entity graph plus community summaries; aimed at whole-corpus questions, unlike our fact-lookup questions |
| 4 | B. Jiménez Gutiérrez *et al.*, "HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models," NeurIPS 2024. | arXiv:2405.14831 | GraphRAG, multi-hop | Knowledge graph + Personalized PageRank for multi-hop retrieval; the closest "graph helps multi-hop" claim to test on our multi-hop questions |
| 5 | A. Asai *et al.*, "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection," ICLR 2024. | arXiv:2310.11511 | Adaptive RAG | Model decides when to retrieve and critiques its own output; contrast with our fixed pipeline and external NLI check |
| 6 | S.-Q. Yan *et al.*, "Corrective Retrieval Augmented Generation," 2024. | arXiv:2401.15884 (preprint) | Robust RAG | Evaluates retrieval quality and corrects it; related to our unanswerable / poor-retrieval cases |
| 7 | J. Chen *et al.*, "Benchmarking Large Language Models in Retrieval-Augmented Generation," AAAI 2024. | arXiv:2309.01431; DOI 10.1609/AAAI.V38I16.29728 | RAG evaluation | RGB benchmark; "negative rejection" (answer not in the context) is exactly our unanswerable category |
| 8 | S. Es *et al.*, "RAGAs: Automated Evaluation of Retrieval Augmented Generation," EACL 2024 (demos). | arXiv:2309.15217 | RAG evaluation | Reference-free faithfulness / relevance metrics; we compare our NLI trust score with this style of metric |
| 9 | J. Saad-Falcon *et al.*, "ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems," NAACL 2024. | arXiv:2311.09476 | RAG evaluation | Automatic judges calibrated with a small human-labelled set; supports our human-validation step |
| 10 | S. Min *et al.*, "FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation," EMNLP 2023. | arXiv:2305.14251 | Hallucination metrics | Share of atomic facts supported by a source; the idea behind our claim-level hallucination rate (ours splits at sentence level) |
| 11 | P. Manakul, A. Liusie, M. J. F. Gales, "SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models," EMNLP 2023. | arXiv:2303.08896; ACL 2023.emnlp-main.557 | Hallucination detection | Checks consistency across sampled answers; a cheap extra signal we could add |
| 12 | J. Li *et al.*, "HaluEval: A Large-Scale Hallucination Evaluation Benchmark for Large Language Models," EMNLP 2023. | arXiv:2305.11747; ACL 2023.emnlp-main.397 | Hallucination benchmark | Shows models struggle to recognise hallucinations; giving external knowledge helps |
| 13 | L. Huang *et al.*, "A Survey on Hallucination in Large Language Models: Principles, Taxonomy, Challenges, and Open Questions," ACM TOIS, 2024. | arXiv:2311.05232 | Hallucination survey | Source for our intrinsic vs extrinsic definition of hallucination |
| 14 | N. F. Liu *et al.*, "Lost in the Middle: How Language Models Use Long Contexts," TACL, 2024. | arXiv:2307.03172 | Context use | Models use the start and end of a long context best; relevant to how many passages (top-k) we feed a 4B model |

## Gap statement (for Section 2.3)

Graph-based RAG is usually evaluated on large general corpora with
automatically built graphs, and rarely with questions split by type or with
the "answer not in the corpus" case measured. We test vector, graph and hybrid
retrieval against a closed-book baseline on a single history book, on
questions written independently of the retrieval system, with unanswerable
and multi-hop questions reported separately, and with a human check of the
automatic hallucination scorer.

## Still to do by hand

- Open each paper and fill the report's table columns (methodology, dataset and
  metrics, key numbers, limitations) from the paper itself.
- The review asks for a search strategy (databases, keywords, dates, counts).
  Record what you actually searched; the list above is only what was verified.
