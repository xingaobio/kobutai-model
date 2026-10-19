## Problem Statement

A desktop companion application (a browser extension, with planned macOS and Windows versions) shows an animated character that reacts to whatever media the user is playing. To choose the character it must decide, for every track or video, (a) the content type: music, video, live stream or podcast; (b) for music, one of ten coarse genres: rock, western-pop (including K-pop and R&B), jpop, anime (including Vocaloid, VTuber and game music), cpop, electronic, lofi (including ambient and sleep sounds), classical (including film scores), hiphop, other (jazz, country, folk, enka, children's songs and the rest); and (c) for video, whether it is entertainment (ent: memes, gaming, vlogs, variety) or informational (info: education, news, science, how-to).

Constraints that define the problem:

- **Text only.** The only available signal is metadata: the title, the artist or channel name, the website host name, and sometimes the duration in seconds. Audio is not available (capturing system audio needs extra permissions and power).
- **Trilingual and mixed script.** Titles are in English, Japanese and Chinese, often mixed in one string (e.g. Japanese title with an English "Official Music Video" suffix, full-width characters, emoji).
- **Fully on-device and cross-platform.** The same model must run in a browser extension service worker (TypeScript), in Swift and in C#, with bit-identical outputs, fit comfortably in an extension package, and take well under a few milliseconds per item.
- **Abstention is allowed.** If unsure, the system may return "no answer" and the application falls back to a generic character; a wrong answer only means a mismatched character.

The research question: how far can a tiny text-only on-device model go on this task, what limits it, and does a larger model (via distillation or a small multilingual transformer) remove that limit?

## Core Hypothesis

Most of the genre signal in a media title is *knowledge* (who the artist is, which anime a song belongs to, what a meme is) rather than linguistic structure. Therefore (1) a very small hashed character n-gram linear model, combined with deterministic site rules and an updatable artist-to-genre lookup table, captures nearly everything that can be learned from surface form; (2) making the on-device learner stronger (distilling a large language model's labels into it, or replacing it with a ~100M-parameter multilingual transformer) will not close the gap to a large language model, because the residual errors are missing world knowledge, not missing capacity; and (3) the right design is a confidence-gated hybrid: answer locally when confident, and hand only uncertain items to a large language model.

A secondary hypothesis about evaluation: validation sets written by the same large language model that wrote the synthetic training data overstate accuracy; a test set of real titles scraped from live charts and trending pages is necessary to measure progress.

## Proposed Methodology (Detailed Technical Approach)

**Pipeline.** input (title, artist, site, duration) → text normalisation → (1) site rules → (2) artist table bonus → (3) linear model → (4) confidence thresholds → label or abstain; optionally (5) hand uncertain items to a large language model.

**Normalisation.** Unicode NFKD; drop combining marks U+0300–U+036F; lower-case ASCII letters only (so behaviour is identical across runtimes); map tabs, newlines, ideographic space to space; collapse whitespace. Site: normalised host with leading `www.` or `m.` removed, plus an alias table.

**Features.** For the title and the artist separately, take the first 160 Unicode code points, wrap with boundary markers `^` and `$`, and emit character n-grams for n = 1..4 tagged by field (`t`, `a`), plus untagged copies for n ≥ 2 (shared between fields). Also emit ASCII word unigrams and adjacent word bigrams (words = runs of [a-z0-9]). Context features: site, duration bucket, and site × duration bucket. Duration buckets: U (unknown), L (live/infinite), a < 60 s, b < 150 s, c < 330 s, d < 600 s, e < 1500 s, f < 3600 s, g ≥ 3600 s.

**Hashing.** Each feature string is hashed with 32-bit FNV-1a over its UTF-8 bytes and masked to $B = 2^{17} = 131{,}072$ buckets. The input is the set of distinct bucket indices $S$; features are binary and scaled by $1/\sqrt{|S|}$.

**Model.** Three multinomial logistic-regression heads share the hashed input: type ($K=4$), genre ($K=10$, trained on music rows only) and video class ($K=2$, trained on video rows only). For head $h$ with weights $W_h \in \mathbb{R}^{K_h \times B}$ and bias $b_h$:
$$z_h = b_h + \frac{1}{\sqrt{|S|}} \sum_{i \in S} W_h[:, i], \qquad p_h = \mathrm{softmax}(z_h).$$
Training: scikit-learn lbfgs, $C = 30$, balanced class weights. Augmentation: each row is duplicated once with the artist removed (35% chance), duration set to unknown (50%) or site removed (15%). The genre head is also trained on artist-only and title-only rows built from the artist table.

**Quantisation and format.** Each head is quantised to symmetric int8 with one scale per head ($\text{scale}_h = \max|W_h| / 127$); biases stay float32. A small binary format stores magic, version, bucket count, per-head (K, scale, bias) and an int8 matrix laid out [bucket][16 outputs]. Inference is lookup-and-sum plus softmax: tens of lines in each language. A shared file of test vectors (inputs with expected outputs to 1e-9) enforces identical behaviour across Python, TypeScript, Swift and C#.

**Site rules.** Music-only sites force type = music; podcast sites force podcast; film and course sites restrict the type softmax to video; live streams (duration infinite or platform flag) restrict it to {music, live}. A platform-provided category (e.g. a podcast episode URL) overrides the model.

**Artist table.** A map from normalised artist name or alias to genre. If the artist field (or one of its segments split on separators such as "/", "&", "feat") matches, add a bonus $\beta_a$ to that genre's logit; otherwise if a title segment matches, add $\beta_t$. A bonus rather than an override keeps contextual cues in the title (e.g. "TV anime OP") able to change the answer. Conflicts between sources are resolved by priority: curated lists > chart-derived labels > training-data votes.

**Thresholds and confidence.** A head abstains below its threshold (type 0.4, genre 0.3, video class 0.5 in the shipped model, tuned on a real development split). Combined confidence for hand-off is $\min(p_{\text{type}}, p_{\text{genre}})$ for music, $\min(p_{\text{type}}, p_{\text{vclass}})$ for video, $p_{\text{type}}$ otherwise, and 0 on any abstention.

**Large-model teacher and hybrid.** A large language model (DeepSeek V4.1 Flash, self-hosted, temperature 0, JSON output, 20 titles per request) receives exactly the same four fields and the same label definitions. Used three ways: (i) as an upper-bound reference; (ii) as a teacher for distillation (labelling unlabelled real titles and auditing existing labels); (iii) in a hybrid where items with combined confidence below $\tau$ are sent to it.

**Transformer comparison.** mMiniLMv2-L6-H384 (multilingual, distilled from XLM-R Large; 107M parameters, of which 96M are the vocabulary embeddings), input "title | artist | site | duration bucket", mean pooling, three linear heads; same site rules and artist bonus as the linear model.

**Evaluation protocol.** Main metric "fully right": the type is correct and, for music, the genre is correct, and for video, the video class is correct. Reported without thresholds (the model must always answer) so that all models are compared on equal terms, plus coverage-aware metrics for the shipped thresholds. The real-world set is split into development and test parts by a fixed hash of "artist|title"; the test part is used only for reporting.

## Expected Contribution

1. A practical recipe and open artefacts for an on-device, trilingual (EN/JA/ZH), text-only media classifier of about 2 MB that runs in under 1 ms and gives identical outputs across four runtimes.
2. Evidence, from two controlled negative results, that the residual gap to a large language model on this task is a knowledge gap: neither distillation of LLM labels nor a 50–200× larger multilingual transformer improved real-world accuracy.
3. A confidence-gated hybrid that hands about a third of items to a large language model and matches or slightly exceeds sending everything to it.
4. A cautionary measurement of how much synthetic (LLM-written) validation data overstates accuracy compared with real scraped titles, and an ablation of a licence-clean variant trained without chart data.
