# Fact-check: drafts/paper.tex (iter1)

Ground truth: `inputs/experimental_log.md` (LOG), `kobutai-classifier-research/results/*.csv` (CSV), `REPORT.md`, `TIMELINE.md`, licence-clean `hf/model/eval/results.json` (RJ). Method details not in LOG were checked against `kobutai-classifier/spec/SPEC.md`, `train/train.py` and `scripts/collect_results.py`. Line numbers refer to `drafts/paper.tex`.

## Summary

- **Numeric value mismatches: 0.** Every number in the abstract, text, Tables 1-4, figure captions and figure images matches LOG/CSV/RJ after rounding. All 7 Wilson intervals and both McNemar p-values were recomputed from the counts and reproduce exactly.
- **Factual errors (statements contradicted by the inputs or by the paper's own figures/data): 10** (E1-E10 below).
- **Unsupported or over-stated claims (not contradicted, but not backed by the inputs): 8** (U1-U8).
- **Citation mis-descriptions: 2** (C1, C2). All other cited-work statements are consistent with the titles/abstracts in citation_pool.json.

---

## 1. Factual errors

| # | Line | Paper says | Correct | Source |
|---|---|---|---|---|
| E1 | 27 (abstract) | "gives identical outputs across runtimes" | Verified only for Python and TypeScript (559/559). Swift and C# were "written but not compiled or run". The paper itself says so at L45 and L280. | LOG 2.1; REPORT 8; TIMELINE 01:45 |
| E2 | 97 | "A shared file of test vectors ... keeps the Python reference, TypeScript, Swift and C# runtimes identical." | Same as E1: identity is verified only for TypeScript. | LOG 2.1 |
| E3 | Fig. 1 image (`fig_pipeline_overview.png`, footer) | "identical outputs in Python, TypeScript, Swift and C#" | Same as E1. The figure image must be regenerated. | LOG 2.1 |
| E4 | 269 | "from round 2 on, every decision was made on the real development split and the real test split was used only for reporting." | (a) The adoption decisions in Table 3 (distillation not adopted) and Sec. 6.3 (curated override adopted, 2^18 not adopted, round-5 data adopted) were made on "real dev + real test combined, 637 rows". (b) tau = 0.9 was chosen while reading the real-test result ("置信度 < 0.9 时转交 32%，真实测试 95.6%"). The real-dev curve peaks at tau = 0.95 (97.2%). (c) The round-3 diagnosis was triggered by the drop on the real test set. LOG 1 also claims "the test part only for reporting", so the inputs are internally inconsistent; the paper inherits the inconsistency and should disclose the actual protocol. | LOG 2.8, 2.9 headers; TIMELINE line 24; LOG 2.6; REPORT 4 round 3 |
| E5 | 162 (Table 2 caption) | "Fully right (%) on every evaluation set in the unified setting (no thresholds, same artist bonus)" | The hybrid row is not in the unified setting. It uses the shipped linear model with its thresholds and title bonus (+3), plus DeepSeek below tau. | LOG 1 "Hybrid: shipped linear model with its thresholds"; LOG 2.6 note; collect_results.py L99-101 |
| E6 | 194 (Fig. 3 caption) | "The leftmost point is the shipped model alone with its thresholds, where abstentions count as errors (84.1%)" | The 84.1 vs 84.9 gap is 3 items. Abstentions are 0.5% = 2 items. LOG attributes the difference to thresholds **and** the +3 title bonus of the shipped configuration. The Appendix caption (L301) says this correctly; Fig. 3's caption omits the title bonus. | LOG 2.6, paragraph after table |
| E7 | Fig. 3 image (`fig_hybrid_handoff_curve.png`) | Annotation "tau=.9" placed at about x = 21-25% sent, above the tau = 0.8 (21.4%) and tau = 0.85 (25.3%) markers | The tau = 0.9 point is at x = 31.9%, y = 95.6. The "tau=.7" label is also offset to the right of its point (16.7%). | CSV hybrid_curve.csv |
| E8 | 188 (Fig. 2 caption), 201, 285 | "A 200-times larger multilingual encoder"; "The transformer, 200 times larger"; "a 200-times larger multilingual encoder" | 200x compares the fp32 transformer file (428 MB) with the int8 linear file (2.1 MB). Like for like, the ratio is about 51x: int8 107 MB / 2.1 MB = 51, and 107M parameters / (16 x 2^17 = 2.1M) weights = 51. L46 and REPORT use the correct "50 to 200 times". Use that range or "about 50 times (int8)". | LOG 2.1; REPORT 摘要 ("大 50～200 倍") |
| E9 | 95 | "Every training row is duplicated once with the artist removed (probability 0.35), the duration set to unknown (0.5) or the site removed (0.15)" | The three drops are applied **independently** to one copy. The copy is kept only if at least one drop fired, and the duration drop applies only when the duration is known and not live. "or" implies the drops are mutually exclusive. Suggested wording: "...a copy in which, independently, the artist is removed (0.35), a known duration is set to unknown (0.5) and the site is removed (0.15)". | LOG 1 "augmentation copy per row"; train.py `augment()` L39-54 |
| E10 | 257 | "If the gap were a matter of decision boundaries, knowledge distillation \citep{hinton2015distilling} should transfer..." | The experiment adds **hard** teacher labels filtered at confidence >= 0.8, plus relabelling or dropping of rows. That is pseudo-labelling or label distillation, not Hinton-style soft-target distillation. The negative result does not cover soft-target KD. See also C1. | LOG 2.8; citation_pool hinton2015distilling |

## 2. Unsupported or over-stated claims (not in experimental_log.md / REPORT.md)

| # | Line | Claim | Status |
|---|---|---|---|
| U1 | 45 | "openly released artefacts" | Not in LOG or REPORT. A public release does exist (`kobutai-model/README.md` links GitHub and Hugging Face `xingaouk/kobutai-media-classifier`). The HF model is the **licence-clean** variant (79.9%, CC-BY-NC-4.0), not the 84.9% shipped model. Give the URL(s), say which model and licence, and reconcile with "Anonymous" as author. |
| U2 | 55 | "Our searches found little work on genre classification from short titles in mixed English, Japanese and Chinese." | No literature search is recorded in the inputs. Rephrase as "We are not aware of work on..." or drop. |
| U3 | 35 | "Our setting is a desktop companion application that shows an animated character" | REPORT describes Kobutai as a browser extension with shared models for browser, macOS and Windows. Suggested wording: "a browser extension (with desktop versions) that shows...". |
| U4 | 111 | "so any gain over the small model comes from what the large model knows, not from extra input" | Equal input rules out extra input only. The gain could also come from capacity or language understanding, which is the hypothesis under test. Over-claim. |
| U5 | 206 (also 47, 194) | "The hybrid can beat the large model alone because..." | 95.6 vs 95.3 = 366 vs 365 items (1 item), with overlapping Wilson intervals (93.0-97.2 vs 92.7-97.0). The data support "matches". |
| U6 | 231 | "Targeted data does help where surface patterns exist." | Measured only on the LLM-written focus set. REPORT 5.4: "专项集也是大模型凭记忆写的，绝对值偏乐观，但前后对比可信". This caveat is missing, and it matters given Sec. 6.3's own argument about synthetic evaluation. |
| U7 | 104 / 149 | Artist bonus beta_a, beta_t: only the unified values (3, 0) are given | Omission. The shipped model, the hybrid and Table 4 use beta_a = beta_t = 3 (model/config.json `artistBonus`; LOG 2.6 note). The licence-clean variant uses (3, 0) (LOG 1). |
| U8 | 201 | "the linear model takes 0.85-0.97 ms per item in Node.js; the int8 transformer takes 6.6 ms" | Both numbers are correct (LOG 2.1), but they come from different runtimes (Node.js vs ONNX Runtime). Name the runtimes when comparing. |

**Method details not in LOG/REPORT but verified correct against code/spec** (no action needed): U+0300-U+036F drop, ASCII-only lower-casing, ideographic space (U+3000), `www.`/`m.` stripping and alias table (L86; SPEC 2). Untagged copies for n >= 2, word unigrams/bigrams, seven duration ranges <60 s ... >= 3600 s (L88; SPEC 2-3). The 1/sqrt(|S|) scaling in Eq. 1 (SPEC 4). Artist-only and title-only rows for the genre head (L95; train.py L86-91). Segment splitting on "/", "&", "feat" (L104; SPEC 6). Combined confidence c = min(...) with c = 0 on abstention (L106; collect_results.py L103-110). Site-rule behaviour (L102; SPEC 5).

## 3. Numbers verified (all match)

- **Abstract (L27):** 2.1 MB; <1 ms; 383; 84.9; 95.3; 107M; 84.1; 31.9; 95.6; 10.5-15.3. OK.
- **Intro (L35-48):** 2.1 MB int8; <1 ms Node.js 2-core; 559/559; 383; 84.9; 95.3; 107M; "50 to 200 times"; 84.1; 31.9; 95.6; 95.3; 10.5-15.3; licence-clean 5.0 points (84.9 -> 79.9); p = 0.0026. OK.
- **Related work (L55-64):** 117 genres; 45%; tau = 0.9 -> 31.9%. OK.
- **Method (L81-111):** 160 code points; n = 1..4; 2^17 = 131,072; K = 4/10/2; C = 30; 0.35/0.5/0.15; max|W|/127; 2,097,256 bytes; 1e-9; 25,826 keys; thresholds 0.4/0.3/0.5; temperature 0, 20 per request. OK.
- **Data (L119, Table 1, L144):** 8,290 songs, 12 storefronts, 29 charts; 716 removed; 21,377; 21,780 -> 25,826; 7,673 / 2,511 / 3,044 / 8,290 / 361 / 214, which sum to 22,093 as in LOG; 401/562; EN 219 / JA 212 / ZH 206; 254/383; 360; 963 (~320 per language); 637; 360/152/48/77 (also matches data_inventory.csv); 214 test music. OK.
- **Setup (L149-151):** beta 3/0; 107M / 96M / 11M; lr 6e-5, batch 32, 3 epochs, 37 min; 13,787 keys. OK.
- **Table 2 (L170-178):** all 64 cells match LOG 2.2 and model_comparison.csv; the licence-clean row matches RJ (0.799 / 0.9167 / 0.9585 / 0.9452 / 0.7383 / 0.8191). OK.
- **Sec. 5.1 (L199-201):** 84.9 [80.9, 88.1]; 95.6 / 81.3 / 86.2; 95.3 [92.7, 97.0]; 47 vs 7, p < 0.0001 (exact p = 2.3e-8); 84.1 [80.1, 87.4]; 86.7 [82.9, 89.7]; +1.8 = 6.97 items, i.e. about 7; +3 bar; 83.1 vs 90.3; 84.7 vs 94.9; 0.85-0.97 ms; 6.6 ms; 107 MB; 559/559. OK.
- **Sec. 5.2 (L206):** 31.9; 95.6 [93.0, 97.2]; 95.3; 26.8 / 96.9; 20.0 / 93.6; 18.0 / 99.0; licence-clean 94.8 with 44.1% sent (RJ 0.9478, 0.4413). OK.
- **Sec. 5.3 (L229-231) and Fig. 4:** anime 58.6 / 79.3; lofi 46.7 / 80.0; hiphop 61.1 / 83.3; other 68.2 / 100; confusions anime->jpop 5, anime->cpop 3, other->not music 5, lofi->rock 3; 63 of ~19,000; 1,420 / 717 / 212 / 907 / 352; 75.8->89.2, 85.0->90.8, 88.3->90.8; DeepSeek 93.3 / 96.7 / 91.7. All Fig. 4 bars, including the green licence-clean bars, match LOG 2.5 and RJ genre_recall. The bar order is by gap, as the caption says. OK.
- **Table 3 / Sec. 6.1 (L248-259):** all 15 cells; 16,299; 3,845; 640; 8,982; >= 0.8; 21,535; 286; >= 0.9; 81.9-83.3; 87.5 -> 79.6; 600 / 1,500; ~150 videos; 89 / 13,245 (0.7%); 197 / 8,290 (2.4%). OK.
- **Sec. 6.2 (L264):** 84.1; 428 / 107 / 2.1 MB; 84.6 -> 79.9. OK.
- **Sec. 6.3 (L269-275):** 97.5 / 96.7; 46 rows, 91.1 (these 46 rows are from synthetic dev+test, not the test split as the sentence implies; minor); 81.4; 10.5-15.3; 78.6 vs 76.1; 254; 471 -> 72; 82.8; 96.7 -> 92.8; 84.9 -> 79.9 [75.6, 83.6]; 81.3 -> 73.8; 28 vs 9, p = 0.0026 (exact 0.00256); 91.7; 95.9; 2^18, 4.19 MB, ~2 points (relative to the pre-override round-4 model, 78.6 -> 80.8). OK.
- **Fig. 2 image:** 95.6 / 95.3 / 86.7 / 84.9 / 84.1 / 79.9 / 77.8, intervals and sizes. OK.
- **Fig. 3 image:** all 9 points match hybrid_curve.csv; the 95.3 and 84.9 reference lines are correct. Annotation placement is wrong (E7).
- **Fig. 5 image:** 96.7 / 81.4 / 15.3; 97.7 / 84.1 / 13.6; 96.0 / 85.5 / 10.5; legend n = 562 / 383. OK.
- **Limitations (L280):** 383 / 214 / ~100 video / <2 points / 14-29 items. OK.
- **Conclusion (L285):** 84.9; 95.3; 95.6; "10 to 15 points" (rounded 10.5-15.3). OK, apart from "200-times" (E8).
- **Disclosure (L289):** seven rounds in one day. OK (LOG 1; REPORT header).
- **Table 4 (L307-312):** all 30 cells match LOG 2.6 and hybrid_curve.csv; "0.5% abstentions". OK.

## 4. Statements about cited work (vs citation_pool.json)

| Key | Line | Paper's statement | Verdict |
|---|---|---|---|
| tzanetakis2002musical | 33, 55 | genre recognition studied mainly from audio | OK (title) |
| schedl2014music | 33 | audio plus web pages and collaborative tags | OK (abstract) |
| joulin2017bag | 37, 58 | on par with deep classifiers; many orders of magnitude faster to train and evaluate | OK (near-verbatim) |
| weinberger2009feature | 37, 58, 88 | fixed buckets, "bounded effect of collisions" | OK. The abstract gives exponential tail bounds and negligible interaction. |
| joulin2016fasttextzip | 37, 58 | product quantisation; about two orders of magnitude less memory; small accuracy loss | OK |
| gholami2021survey | 37, 58 | quantisation / post-training integer quantisation | OK (survey) |
| gilardi2023chatgpt | 37, 61 | LLMs label text with little training; "match or exceed crowd workers" | OK. Understated: the abstract says it outperforms by ~25 points. |
| wang2021want | 37, 61 | label with little task-specific training; LLM labels train small models | OK (GPT-3 as few-shot labeler) |
| sun2023text | 37 | careful prompting may be needed to match fine-tuned classifiers | OK. CARP also uses a fine-tuned model for kNN demonstrations; acceptable. |
| deepseekai2024deepseekv3 | 37, 111 | "DeepSeek V4.1 Flash, a model of the DeepSeek family" | **C2 (loose).** The V3 report does not describe V4.1 Flash; readers may assume the 671B/37B-active MoE specs. Say "a successor of DeepSeek-V3" and give the model's own reference or URL if one exists. |
| li2023synthetic | 37, 61 | synthetic data usefulness falls as subjectivity rises | OK |
| long2024llmsdriven | 37, 61 | LLMs can write labelled training data; synthetic sets as a route | OK (survey) |
| mayer2008rhyme | 55 | rhyme and stylistic lyric features | OK (title) |
| fell2014lyrics | 55 | lyrics analysis combined with classification | OK (title) |
| tsaptsinos2017lyrics | 55 | HAN, up to 117 genres | OK |
| oramas2017multilabel | 55 | audio, reviews, cover images, multi-label | OK |
| lamere2008social | 55 | social tags | OK (title) |
| knees2004artist | 55 | web-based data about artists | OK (title) |
| filippova2011improved | 55 | video categories from text metadata and comments | OK (title) |
| wang2012baselines | 58 | linear n-gram models are strong sentiment/topic baselines | OK (title) |
| bojanowski2017enriching | 58 | subword char n-grams help rare words | OK |
| peng2003language | 58 | char n-gram LMs for Chinese, Japanese, English, Greek without preprocessing | OK |
| zhang2015character | 58 | char-level ConvNets competitive on large datasets | OK |
| liu1989limited, pedregosa2011scikitlearn | 58, 95 | L-BFGS; scikit-learn | OK |
| conneau2020unsupervised | 61, 151 | XLM-R multilingual encoder | OK |
| hinton2015distilling | 61, 257 | L61: "compresses what a large model or ensemble has learned into a smaller model" | OK at L61. **C1 at L257:** used to motivate a hard-label pseudo-labelling experiment (see E10). |
| sanh2019distilbert, wang2020minilm | 61 | distilled compact encoders | OK |
| wang2021minilmv2 | 61, 151 | multilingual MiniLMv2 distilled from XLM-R Large | OK. The abstract lists XLM-R base and large teachers; the specific checkpoint is documented in REPORT 6.2. |
| abdaoui2020load | 61 | most parameters in embeddings; trimming removes up to 45% | OK |
| tang2019distilling | 61 | single-layer BiLSTM from BERT, competitive with larger models | OK (comparable to ELMo with ~100x fewer parameters) |
| hsieh2023distilling | 61 | teacher labels or rationales train small models | OK |
| loshchilov2017decoupled | 151 | AdamW | OK |
| chow1970optimum | 64 | reject option trades coverage for error | OK |
| geifman2017selective | 64 | selective classification with a guaranteed risk level | OK ("with high probability" omitted; acceptable) |
| guo2017calibration | 64 | "works only as well as the underlying confidence estimates, which modern networks often get wrong" | OK. The second half is from the abstract; the first half is the authors' inference and is phrased as such. |
| varshney2022model | 64 | cascading reduces computation and "in some settings" improves accuracy | OK. Understated: the abstract says it consistently improves accuracy. |
| chen2023frugalgpt | 64 | learns which combination of LLMs to query; lower cost with kept or improved accuracy | OK |
| ding2024hybrid | 64 | routes each query to a small edge or large cloud model by predicted difficulty | OK |
| wilson1927probable, mcnemar1947note | 149 | Wilson interval; McNemar test | OK |

All 42 cited keys exist in refs.bib, and the PDF has no unresolved references.
