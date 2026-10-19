## §1 · Core Claim and Narrative
_Written by: outline-agent, Step 1_

**Core claim:** A 2.1 MB hashed n-gram linear model reaches 84.9% fully right on real trilingual media titles; neither LLM distillation nor a 107M-parameter multilingual encoder improves it, because the residual errors are missing world knowledge; a confidence-gated hand-off of ~32% of items to an LLM reaches 95.6%.
**Narrative tension:** On-device constraints push toward tiny models, but genre depends on knowing the artist — does a stronger on-device learner help?
**Key novelty framing:** An empirical, negative-results-heavy case study with open artefacts, not a new algorithm.
**Outline decisions:**
- Plotting plan: 5 figures (1 diagram, 4 plots)
- Related Work clusters: text/metadata music classification; hashed linear models; compact multilingual transformers + distillation; cascades/selective prediction
- Section structure: Abstract, Introduction, Related Work, Method, Experimental Setup, Results, Analysis and Negative Results, Limitations, Conclusion, Disclosure
**Potential weaknesses flagged at outline stage:**
- Real test set has 383 items with LLM-produced labels; teacher agreement may be inflated.
- Single-day, single-task study; generalisation claims must stay narrow.

## §3 · Literature context
_Written by: literature-review-agent, Step 3. Pool: 42 verified references (citation_pool.json, refs.bib); 39/42 cited in drafts/intro_relwork.tex. Cutoff 2026-10-01._

**How prior work positions this paper**
- Genre recognition is mainly studied from audio (tzanetakis2002musical) and from rich context such as web pages and tags (schedl2014music, lamere2008social, knees2004artist). Text-based work relies on full lyrics (mayer2008rhyme, fell2014lyrics, tsaptsinos2017lyrics), reviews (oramas2017multilabel) or comments (filippova2011improved). None of it uses only title/artist/site/duration, and our searches found little on mixed English/Japanese/Chinese titles. The paper says "our searches found little work", not "first".
- Hashed n-gram linear models are an established, strong, small baseline (joulin2017bag, wang2012baselines, weinberger2009feature, joulin2016fasttextzip). Character n-grams handle Chinese and Japanese without segmentation (peng2003language, zhang2015character, bojanowski2017enriching). The contribution is the end-to-end recipe (rules + artist table + int8 + cross-runtime test vectors) and the measurements, not the model class.
- Multilingual vocabulary embeddings dominate encoder size (abdaoui2020load). This supports the point that 96M of mMiniLMv2's 107M parameters are embeddings.
- Distillation is usually reported as a positive result (hinton2015distilling, sanh2019distilbert, tang2019distilling, hsieh2023distilling), and LLM labels are cheap and often good (wang2021want, gilardi2023chatgpt). Our negative distillation result should be framed as specific to entity-knowledge-bound labels, as a hypothesis, not as a general claim.
- Li et al. (li2023synthetic) found synthetic data helps less as subjectivity rises. Our synthetic-vs-real gap (10.5-15.3 genre points) is consistent with but distinct from this: it concerns evaluation sets, not only training data.
- Cascades and routing (varshney2022model, chen2023frugalgpt, ding2024hybrid) target serving cost, with a language model as the small model. ding2024hybrid explicitly frames the small model as an edge model, so do not claim the edge/cloud split is new. What differs here is the tiny non-neural local model and the privacy/offline motivation. varshney2022model also reports that cascades can beat the largest model alone, which matches our 95.6 vs 95.3 finding; cite it when discussing why the hybrid exceeds "always".
- Selective prediction (chow1970optimum, geifman2017selective) and calibration (guo2017calibration) are the formal framing for thresholds and hand-off. We did not calibrate the linear model, so mention it as a limitation or future work if relevant.
- Do NOT claim to beat any cited system. Only the in-log baselines (linear rounds, mMiniLMv2, DeepSeek, hybrid) are compared.

**Bib keys for each citation hint in outline.json section_plan**
- Intro, "music genre classification survey or paper": tzanetakis2002musical, schedl2014music
- Intro, "fastText": joulin2017bag (plus joulin2016fasttextzip for FastText.zip compression)
- Intro, "feature hashing": weinberger2009feature
- 3.2, "FNV hash": no citation (no paper exists; describe FNV-1a in text only)
- 3.2, "L-BFGS": liu1989limited
- 3.2, "scikit-learn": pedregosa2011scikitlearn
- 3.2, int8 quantisation (optional): gholami2021survey
- 3.4, "DeepSeek-V3 / DeepSeek family": deepseekai2024deepseekv3. No technical report exists for "DeepSeek V4.1 Flash". Cite the V3 report as the closest official DeepSeek technical report, with wording like "a model of the DeepSeek family (DeepSeek V4.1 Flash; see the V3 report for the architecture family)". Do not attribute V4.1 specifics to that report.
- 4.2, "Wilson interval": wilson1927probable (not yet cited; for the Setup section)
- 4.2, "McNemar": mcnemar1947note (not yet cited; for the Setup section)
- 4.2, "AdamW": loshchilov2017decoupled. The verifiable record is arXiv 1711.05101 (2017), so the bib entry is an arXiv preprint dated 2017, not ICLR 2019. Not yet cited; for the Setup section.
- 4.2, "MiniLMv2": wang2021minilmv2 (plus wang2020minilm)
- 4.2, "XLM-R": conneau2020unsupervised
- 4.2, "ONNX Runtime": no citable paper found or verified. Skipped; name the software in text without a citation.
- 6.1, "knowledge distillation (Hinton)": hinton2015distilling (also tang2019distilling, hsieh2023distilling)
- 6.2, small encoders (optional): sanh2019distilbert, abdaoui2020load
- 6.3, synthetic data (optional): li2023synthetic, long2024llmsdriven
- Hybrid / 5.2 (optional): geifman2017selective, varshney2022model, chen2023frugalgpt, ding2024hybrid, guo2017calibration

**Notes for the section writer**
- abdaoui2020load: the publisher's DOI record (Crossref, ACL Anthology deposit) spells the title "Mutililingual BERT", and Semantic Scholar spells it "Mutlilingual". The bib follows the Crossref spelling. Do not "correct" it silently; if the author prefers, change it to "Multilingual" in refs.bib.
- No abstract was retrievable for tzanetakis2002musical, fell2014lyrics, mayer2008rhyme, lamere2008social, filippova2011improved, knees2004artist, wang2012baselines, liu1989limited or wilson1927probable. Their descriptions in the draft are limited to what their titles and venues support.
