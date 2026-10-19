# Fact-check: refinement/iter2/paper.tex

Ground truth: `inputs/experimental_log.md` (LOG, including new 2.11 and 2.12), `kobutai-classifier-research/results/*.csv` (CSV), `REPORT.md`, `TIMELINE.md`, licence-clean `hf/model/eval/results.json` (RJ), `train/train.py`, `kobutai_py/core.py`, `kobutai-model/README.md` (release). Line numbers refer to `iter2/paper.tex`. Main text = abstract to L291 (Conclusion); the PDF's main text ends on p. 10.

In addition to reading the log, I re-ran the shipped model (`kobutai-classifier/model`) on the real test split with `core.py` and the cached DeepSeek answers. This checked the new numbers independently (see Section 4).

## Summary

- **Numeric value mismatches: 0.** Every number in the abstract, text, Tables 1-5, captions and figure images matches LOG/CSV/RJ after rounding.
- **iter1 factual errors E1-E10:** 9 fixed and 1 partly fixed (E4). E4 is now correct in Sec. 6.2 and Limitations, but L142 still says the test split was "used only for reporting".
- **iter1 unsupported claims U1-U8:** 6 fixed and 2 partly fixed (U1 release details, U2 literature search).
- **iter1 citation issues:** C1 fixed. C2 (DeepSeek V4.1 Flash cited to the V3 report) is not fixed.
- **New factual errors: 0** apart from the residual E4 contradiction at L142 (counted under E4).
- **New unsupported or imprecise statements: 4** (N1-N4 below), all minor.

---

## 1. Status of iter1 factual errors

| # | iter1 issue | iter2 text | Status |
|---|---|---|---|
| E1 | Abstract: "identical outputs across runtimes" | L27: "its TypeScript port reproduces the Python reference exactly" | **Fixed** |
| E2 | Sec. 3.2: four runtimes identical | L95: "the TypeScript runtime matches the Python reference on all 559 vectors, while the Swift and C# ports have been written but not yet compiled or run" | **Fixed** |
| E3 | Fig. 1 footer listed four runtimes | Image footer: "Python and TypeScript outputs verified identical" | **Fixed** |
| E4 | "real test split was used only for reporting" | L273 now discloses the protocol correctly: thresholds were tuned on dev only from round 3, the ablations used dev+test, and tau was chosen on the test curve. L284 (Limitations) agrees, and L198 says tau was chosen on the test curve. **But L142 (Sec. 4.1) still reads "383 test rows (214 of them music), used only for reporting."** This contradicts L273, L284 and LOG 2.12. (LOG 1 has the same leftover sentence; LOG 2.12 overrides it.) | **Partly fixed: residual contradiction at L142** |
| E5 | Table 2 caption: all rows unified | L160: "All rows except the hybrid use the unified setting ... The hybrid uses the shipped model with its thresholds and beta_t = 3" | **Fixed** |
| E6 | Fig. 3 caption left out the title bonus | L186: "84.1%: two abstentions count as errors and the title bonus accounts for one more item". I recomputed this: unified 325/383; thresholds only 323; title bonus only 324; both 322 (= 84.1%). The 2 abstained items were both right in the unified setting. | **Fixed (verified)** |
| E7 | "tau=.9" label misplaced | The label now sits next to the (31.9, 95.6) point. The "tau=.7" label is slightly to the lower right of its point, which is acceptable. | **Fixed** |
| E8 | "200 times larger" | L193: "50 to 200 times the linear model's file" (107/2.1 = 51; 428/2.1 = 204). The size figure carrying the old caption was dropped. Conclusion: "much larger". | **Fixed** |
| E9 | Augmentation drops read as exclusive | L93: "independently ... a known duration ... the copy is kept if at least one field changed". This matches `train.py` `augment()` L39-54. | **Fixed** |
| E10 | Hinton KD vs hard labels | L266: "We used hard labels (pseudo-labelling), as in wang2021want, rather than the soft targets of classic distillation, so our result does not rule out the latter." Limitations repeat this. | **Fixed** |

## 2. Status of iter1 unsupported claims

| # | iter1 issue | Status |
|---|---|---|
| U1 | "openly released artefacts" | **Partly fixed.** L291 now says the shipped model, code, LLM-written training data and evaluation scripts are released, the weights and data under a non-commercial licence, together with a licence-clean variant. All of this is consistent with `kobutai-model/README.md`, which says iter5 is the release and the weights are CC BY-NC 4.0. Still missing: (a) any URL or repository name, and (b) the README's "Data note": the 637-title real-world dev/test set, chart data, trending lists and teacher answers are **not** released. As a result, no headline number can be reproduced from the release, and the paper does not say this. |
| U2 | "Our searches found little work..." | **Partly fixed.** It now reads "In our literature search we found little work" (L53). No search protocol is reported. Acceptable, but "We are not aware of..." would be safer. |
| U3 | "desktop companion application" | **Fixed.** L35: "a browser extension with planned desktop versions". |
| U4 | "any gain ... comes from what the large model knows" | **Fixed.** L109: "any advantage must come from what it has learned about the same text". This is neutral between knowledge and capacity. |
| U5 | "hybrid can beat the large model" | **Fixed.** "on par with sending everything (95.3%, one item fewer)". |
| U6 | Targeted data measured on synthetic focus set | **Fixed.** L240: "although we can only measure this on the LLM-written focus set". |
| U7 | Shipped beta not stated | **Fixed.** L102: beta_a = beta_t = 3 (config.json `artistBonus`). |
| U8 | Latency runtimes not named | **Fixed.** L193: Node.js vs ONNX Runtime, same 2-core CPU (LOG 2.12). |

## 3. New unsupported or imprecise statements (introduced by the revision)

| # | Line | Paper says | Issue | Source |
|---|---|---|---|---|
| N1 | L223, Table 3 caption | Known = "in the artist table or in the training rows" | The definition is imprecise. LOG 2.11 says: artist field **or a title segment** matches the table, **or** the artist appears in the **music** training rows. The split is sensitive to the matching rule. My re-implementation gives 163/51 when matching only the full artist string against the table and the training artists, and 166/48 when segments are also matched; LOG has 168/46. Both reproduce the pattern (linear about 91% vs 47-51%; DeepSeek about 95% vs 87-88%). Two further details: in the unified setting beta_t = 0, so the 12 "title-segment" known items get no table bonus, yet they count as known; and the 92.6% / 75.0% / 75.0% sub-breakdown in LOG is not reported. State the exact rule. | LOG 2.11; my recomputation with core.py and the DeepSeek cache |
| N2 | L223 | "close to the large model's 95.2%" | 90.5 vs 95.2 is a 4.7-point gap (8 vs 16 errors out of 168). "Closer to" or "within 5 points of" is more accurate. The paragraph also omits that DeepSeek itself drops 8.2 points on unknown artists (95.2 -> 87.0). Part of the linear model's 42.7-point drop may therefore reflect item difficulty rather than missing knowledge alone. | LOG 2.11 |
| N3 | L191 | "By language, the linear model is fully right on 89.9% of English, 83.0% of Chinese and 81.4% of Japanese titles." | The values are correct (143/159, 78/94, 96/118; recomputed). However, n = 371, not 383: 12 test items have language tag "xx" (66.7%, 8/12) and are silently left out. Table 1's "EN 219 / JA 212 / ZH 206" are per-file counts that sum to 637, so readers will expect the per-language test n to sum to 383. Add the n values or a footnote. | LOG 2.11; recomputed from data/raw/real |
| N4 | L291 | Release statement | See U1: no URL is given, and the real-world evaluation set is not released. | kobutai-model/README.md "Data note" |

Not errors, but noted. The decomposition in the Fig. 2 caption (2 abstentions + 1 title-bonus item) is not stated in LOG; I verified it by recomputation, so it stands. L198, "about 1 s per request of 20 titles", is correct. LOG 2.12 adds that single-item latency was not measured, and the paper could say so.

## 4. Numbers verified in iter2 (all match)

- **Abstract (L27):** 2.1 MB; under 1 ms; 383; 84.9; 95.3; 107M; 84.1; 90.5 / 47.8 / 87.0 (LOG 2.11: 152/168, 22/46, 40/46); 31.9; 95.6; 10.5-15.3. OK.
- **Intro / contributions (L33-46):** 2.1 MB; under 1 ms; 84.9 / 383 / 95.3; 107M; 31.9 / 95.6; 10.5-15.3. OK.
- **Related work (L53-62):** 117 genres; 45%; XLM-R Large teacher; 2.1 MB. OK.
- **Method (L79-109):** 160 code points; n = 1..4; 2^17 = 131,072; K = 4/10/2; C = 30; 0.35/0.5/0.15 independent (train.py); max|W|/127; 2,097,256 bytes; 10^-9; 559; 25,826 keys; beta_a = beta_t = 3 (config.json); thresholds 0.4/0.3/0.5; temperature 0, JSON, 20 per request. OK.
- **Data (L117, Table 1, L142):** 8,290 / 12 storefronts / 29 charts; 716; 21,377; 21,780 -> 25,826; 7,673 / 2,511 / 3,044 / 8,290 / 361 / 214; 401/562; EN 219 / JA 212 / ZH 206 (data_inventory.csv); 254/383; 360; 963 (about 320 per language); 637; 360/152/48/77; 214 test music. OK. (L142 "used only for reporting" is wrong: E4 residual.)
- **Setup (L147-149):** beta 3/0; 107M / 96M / 11M; 6e-5, batch 32, 3 epochs, 37 min, 2-core CPU; 13,787 keys. OK.
- **Table 2 (L168-176):** all 64 cells match LOG 2.2 / model_comparison.csv; the licence-clean row matches RJ always_answer (0.799 / 0.9452 / 0.7383 / 0.8191; focus 0.9167). OK.
- **Sec. 5.1 (L191-193):** 84.9 [80.9, 88.1]; 95.6 / 81.3 / 86.2; 95.3 [92.7, 97.0]; 47 vs 7, p < 0.0001; 89.9 / 83.0 / 81.4 (recomputed); 428 / 107 MB; "50 to 200 times"; 84.1 [80.1, 87.4]; 86.7 [82.9, 89.7]; +1.8 = about 7 items; +3 bar; 83.1 vs 90.3; 84.7 vs 94.9; 84.6 -> 79.9; 0.85-0.97 ms Node.js; 6.6 ms ONNX Runtime. OK.
- **Sec. 5.2 (L198):** 31.9; 95.6 [93.0, 97.2]; 95.3, one item fewer (366 vs 365); tau chosen on the test curve; dev 0.95 -> 97.2 vs 96.9 (LOG 2.12); about 1 s per 20 titles; licence-clean 94.8 / 44.1 (RJ 0.9478 / 0.4413). OK.
- **Fig. 2 (hybrid) image and caption:** all 9 points match hybrid_curve.csv; 95.3 / 84.9 reference lines; 84.1 decomposition verified (Section 1, E6). OK.
- **Sec. 5.3 (L221-240), Fig. 3, Table 3:** anime 58.6/79.3; lofi 46.7/80.0; hiphop 61.1/83.3; other 68.2/100; confusions 5/3/5/3; Table 3: 168 / 90.5 / 95.2, 46 / 47.8 / 87.0; 21% (46/214 = 21.5%); 24 of 40 genre errors (16 + 24 = 40; 174/214 = 81.3% agrees with Table 2). 63 of about 19,000; 1,420 / 717 / 212 / 907 / 352; 75.8 -> 89.2, 85.0 -> 90.8, 88.3 -> 90.8; 93.3 / 96.7 / 91.7. Fig. 3 bars match LOG 2.5 and RJ genre_recall, and the order is by gap (lofi 33.3 > other 31.8 > hiphop 22.2 > anime 20.7 > western-pop 7.7 > ...), as the caption says. OK.
- **Table 4 / Sec. 6.1 (L257-268):** all 15 cells; 16,299; 3,845; 640; 8,982; >= 0.8; 21,535; 286; >= 0.9; 81.9-83.3; 87.5 -> 79.6; 600 / 1,500; about 150 videos; 89 / 13,245 (0.7%); 197 / 8,290 (2.4%). OK.
- **Sec. 6.2 (L273-279) and Fig. 4:** 97.5 / 96.7; 46 synthetic rows, 91.1; 81.4; 10.5-15.3; 78.6 vs 76.1 (ablations_recorded.csv, bonus 3); 254; 471 -> 72; 82.8; 84.9 -> 79.9 [75.6, 83.6]; 81.3 -> 73.8; 28 vs 9, p = 0.0026; 91.7; 95.9; 2^18, 4.19 MB, about +2 (80.8 vs 78.6 pre-override, dev+test; the baseline is not named). Fig. 4: 96.7/81.4/15.3, 97.7/84.1/13.6, 96.0/85.5/10.5; n = 562 / 383. OK.
- **Limitations (L284):** 383 / 214 / about 100 video / under 2 points / 14-29 per genre. OK.
- **Conclusion (L289-291):** 84.9; 95.3; about a third; "10 to 15 points". OK.
- **Disclosure (L295):** seven rounds, one day. OK.
- **Table 5 (L313-318):** all 30 cells match LOG 2.6; 0.5% abstentions. OK.

## 5. Statements about cited work

Unchanged statements were verified in iter1 and remain OK. Changed or new uses:

| Key | Line | Statement | Verdict |
|---|---|---|---|
| hinton2015distilling | L59, L266 | general KD (L59); "soft targets of classic distillation" (L266) | **OK. C1 fixed.** |
| wang2021want | L266 | "hard labels (pseudo-labelling), as in wang2021want" | OK. The abstract describes GPT-3 "pseudo labels" used to train downstream models. |
| deepseekai2024deepseekv3 | L37, L109 | "DeepSeek V4.1 Flash, a model of the DeepSeek family" cited to the V3 technical report | **C2 not fixed (minor).** The V3 report does not describe V4.1 Flash. Say "a successor of DeepSeek-V3" or cite the model card. |
| wang2021minilmv2, conneau2020unsupervised | L59, L149 | multilingual MiniLMv2 distilled from XLM-R Large | OK |

All cited keys resolve. The PDF has no "??" references.
