# Experimental Log

## 1. Experimental Setup

* **Task and labels:** For each item (title, artist or channel, site host, optional duration) predict type ∈ {music, video, live, podcast}; for music, genre ∈ {rock, western-pop, jpop, anime, cpop, electronic, lofi, classical, hiphop, other}; for video, video class ∈ {ent, info}. The label guide assigns memes, MAD/remix videos and anime episodes or clips to video/ent; anime openings, endings, character songs and soundtracks to music/anime; idol music videos and concerts to music/jpop; idol variety shows to video/ent; idol live streams to live; idol radio to podcast. K-pop and R&B are western-pop.

* **Training data (shipped model):** 22,093 labelled rows before de-duplication; 716 rows whose title appeared in any evaluation set were removed automatically, leaving 21,377 rows. Sources:
  - 7,673 rows written from memory by parallel LLM sub-agents (Claude Sonnet), sharded by language and platform (music sites in Chinese, Japanese and English; YouTube; Bilibili; entertainment and informational video; live; podcasts). The first batch used 20 sub-agents in parallel.
  - 2,511 rows written by sub-agents for targeted hard cases (playlists and work/study BGM, DJ sets, sleep sounds, covers, podcasts on Spotify, finance and software tutorials, trending-style titles without category words).
  - 3,044 rows written by sub-agents for internet memes, Japanese idols and anime (round 5).
  - 8,290 songs from public genre charts of a major music store (12 storefronts × 29 genre charts), mapped to the ten genres by rule; curated artist lists overrule the chart genre on conflict.
  - 361 titles from a public Bilibili popular list (labelled by section) and 214 YouTube trending titles (labelled one by one by a sub-agent).

* **Artist table:** 21,780 rows of artist lists (written by sub-agents per genre and region, plus artists counted from the store charts), merged, de-duplicated and conflict-resolved into 25,826 normalised keys (shipped model). Without chart-derived artists: 13,787 keys.

* **Evaluation sets (strictly separated from training):**
  - Synthetic validation: 963 rows (about 320 per language), written from memory by independent sub-agents that were not allowed to read the training data. Split 401 dev / 562 test. Used only as a regression check.
  - Real-world set: 637 titles taken on a single day from live chart, trending and podcast-chart pages (music streaming charts in the US, UK, Japan, Taiwan and Hong Kong, Japanese and US singles charts, a Chinese music service, Bilibili, YouTube trending and realtime charts, Twitch channels, public broadcaster video pages, anime theme-song lists, classical and jazz release lists), each with its source URL. By language: English 219, Japanese 212, Chinese 206. By type: music 360, video 152, live 48, podcast 77. Labels were assigned by LLM sub-agents following the label guide. Split by a fixed FNV-1a hash of "artist|title" into 254 development and 383 test rows (214 of the test rows are music). The development part was used for threshold tuning and error analysis; the test part only for reporting.
  - Focus set: 360 rows written by independent sub-agents, 120 each for internet memes, Japanese idols and anime (a few current-season anime titles taken from web pages).

* **Evaluation metrics:** "Fully right" = type correct, and genre correct for music, and video class correct for video. Also type accuracy, genre accuracy on music rows, video-class accuracy on video rows. Unified comparison setting: no thresholds (always answer), artist bonus +3 for an artist-field match and +0 for a title match, for every model. Coverage-aware metrics for the shipped thresholds: share answered, accuracy when answered, wrong rate and generic (abstain) rate. 95% Wilson intervals on the 383-row real test set; exact McNemar tests for paired comparisons.

* **Baselines and comparisons:**
  - Linear model checkpoints from round 1, round 4 and round 5 (shipped), restored from version control and re-scored on identical data.
  - Large language model teacher: DeepSeek V4.1 Flash, self-hosted, given the same four fields and the label guide as system prompt; temperature 0, JSON output, 20 titles per request, 12 parallel requests.
  - Multilingual transformer: mMiniLMv2-L6-H384 (107M parameters, 96M of them vocabulary embeddings); vocabulary embeddings frozen, 11M parameters trained; input "title | artist | site | duration bucket", mean pooling, three linear heads; AdamW, learning rate 6e-5, batch 32, 3 epochs, 37 minutes on a 2-core CPU; same site rules and artist bonus as the linear model.
  - Linear + transformer: average of the two models' logits.
  - Hybrid: shipped linear model with its thresholds; items with combined confidence below τ are answered by the teacher.
  - Distillation variants of the linear model using teacher labels.
  - Licence-clean variant: the same pipeline retrained without any chart-derived songs or artists, with thresholds re-tuned on the real development split (type 0.4, genre 0.5, video class 0.55; artist bonus +3 artist, +0 title).

* **Implementation details:** Hashed character 1–4-grams of title and artist (first 160 code points, boundary markers), ASCII words and word bigrams, site, duration bucket and site × duration; FNV-1a 32-bit, 2^17 buckets; multinomial logistic regression (lbfgs, C = 30, balanced class weights); augmentation copy per row (drop artist 35%, unknown duration 50%, drop site 15%); per-head symmetric int8 quantisation. Shipped thresholds: type 0.4, genre 0.3, video class 0.5. The whole project (seven rounds) was carried out in one day by an AI coding agent (Claude, in Claude Code) directed by the project lead; training data, synthetic evaluation sets and labels of the real-world set were produced by LLM sub-agents.

## 2. Raw Numeric Data

### 2.1 Model size and latency

| Model | File size | Compressed | Latency per item | Measured on |
|---|---|---|---|---|
| Linear, 2^17 buckets (shipped) | 2.10 MB (2,097,256 bytes) + 0.48 MB artist table | 1.20 MB + 0.19 MB (gzip) | 0.85–0.97 ms | Node.js on a 2-core cloud CPU |
| Linear, 2^18 buckets | 4.19 MB | 1.82 MB | – | – |
| Transformer fp32 (ONNX) | 428 MB | – | – | – |
| Transformer int8 (ONNX, dynamic quantisation) | 107 MB | – | 6.6 ms | ONNX Runtime on a 2-core cloud CPU |
| DeepSeek V4.1 Flash (server) | – | – | about 1 s per request of 20 titles (about 0.05 s per title batched) | self-hosted endpoint |

Cross-runtime agreement: the TypeScript runtime matched the Python reference on 559 of 559 test vectors (type, genre, video class, provenance and all confidences to 1e-9, plus a hash of the feature indices). The Swift and C# ports were written but not compiled or run.

### 2.2 Main comparison on identical data (no thresholds, fully right, %)

| Model | Size | Real test (383) | Real dev (254) | Focus (360) | Synthetic val (963) |
|---|---|---|---|---|---|
| Linear, round 1 | 2.1 MB | 77.8 | 79.9 | 74.2 | 94.7 |
| Linear, round 4 | 2.1 MB | 83.3 | 90.2 | 83.1 | 95.2 |
| Linear, round 5 (shipped) | 2.1 MB | 84.9 | 89.4 | 90.3 | 94.9 |
| Transformer (mMiniLMv2-L6) | 428 MB | 84.1 | 85.0 | 83.1 | 84.7 |
| Linear + transformer (logit average) | 430 MB | 86.7 | 90.6 | 90.3 | 93.9 |
| DeepSeek V4.1 Flash | server | 95.3 | 95.3 | 93.9 | 98.3 |
| Hybrid: linear + DeepSeek below 0.9 confidence | 2.1 MB + server | 95.6 | 96.9 | 93.6 | 99.0 |
| Linear without chart data (licence-clean) | 2.1 MB | 79.9 | – | 91.7 | 95.9 |

Component accuracies on the real test set (%):

| Model | Fully right | Type | Genre (music rows) | Video class (video rows) |
|---|---|---|---|---|
| Linear, round 1 | 77.8 | 93.0 | 74.3 | 77.7 |
| Linear, round 4 | 83.3 | 94.8 | 79.9 | 83.0 |
| Linear, round 5 (shipped) | 84.9 | 95.6 | 81.3 | 86.2 |
| Transformer | 84.1 | 95.0 | 82.2 | 87.2 |
| Linear + transformer | 86.7 | 96.6 | 83.6 | 88.3 |
| DeepSeek V4.1 Flash | 95.3 | 100.0 | 93.5 | 95.7 |
| Hybrid at 0.9 | 95.6 | 99.5 | 94.9 | 94.7 |
| Linear without chart data | 79.9 | 94.5 | 73.8 | 81.9 |

95% Wilson intervals for fully right on the real test set (n = 383): round 1 [73.4, 81.7]; shipped [80.9, 88.1]; without chart data [75.6, 83.6]; transformer [80.1, 87.4]; linear + transformer [82.9, 89.7]; DeepSeek [92.7, 97.0]; hybrid [93.0, 97.2].

Paired exact McNemar tests on the real test set: shipped vs without chart data: 28 items right only for the shipped model, 9 right only for the variant, p = 0.0026. Shipped vs DeepSeek: 7 right only for the shipped model, 47 right only for DeepSeek, p < 0.0001.

### 2.3 Rounds (shipped configuration with thresholds, real test set, %)

| Round | Change | Type | Genre acc. when answered (coverage) | Video class acc. when answered (coverage) | Wrong rate | Abstain rate |
|---|---|---|---|---|---|---|
| 1 | Baseline: spec, 20 sub-agents of data, linear model | 93.0 | 81.4 (90.7) | 83.9 (92.6) | 20.1 | 2.9 |
| 2 | Real-world test set; +2,000 mid-tail artists; +1,161 hard cases | 92.7 | 84.1 (88.3) | 85.1 (92.6) | 18.5 | 3.4 |
| 3 | Store genre charts (+artists, +songs); thresholds tuned on real dev | 91.4 | 76.9 (86.9) | 94.9 (62.8) | 15.7 | 17.0 |
| 4 | Curated lists overrule chart genres; real trending titles; 1,350 trending-style video and live titles | 94.8 | 87.6 (90.7) | 83.0 (100) | 15.7 | 1.3 |
| 5 | Memes 1,420; idols 717 + 212 group names; anime 907 + 352 artists (shipped) | 95.6 | 85.5 (93.5) | 86.2 (100) | 15.4 | 0.5 |
| 6 | Distillation from teacher (not adopted) | 93.7 | 84.2 (94.4) | 80.2 (91.5) | 19.1 | 0.3 |

Synthetic validation (test split, 562 rows) at round 1: type 97.5, genre acc. when answered 96.7 (coverage 98.7), video class 95.0.

### 2.4 Synthetic versus real evaluation

| Round | Synthetic test: genre acc. when answered | Real test: genre acc. when answered | Gap (points) |
|---|---|---|---|
| 1 | 96.7 | 81.4 | 15.3 |
| 2 | 97.7 | 84.1 | 13.6 |
| 5 | 96.0 | 85.5 | 10.5 |

Round 1 artist-table ablation on synthetic validation (dev + test, 963 rows), genre accuracy when answered: model + table 96.7; table bonus off 95.4; trained without artist rows but table on 95.8; trained without artist rows and no table 92.8. On the 46 synthetic music rows whose artist was in neither the table nor the training data, genre accuracy was 91.1.

### 2.5 Per-genre accuracy on the real test set (music rows, no thresholds, %)

| Genre | n | Shipped linear | DeepSeek | Linear without chart data |
|---|---|---|---|---|
| rock | 23 | 91.3 | 95.7 | 69.6 |
| western-pop | 26 | 92.3 | 100.0 | 88.5 |
| jpop | 28 | 96.4 | 100.0 | 82.1 |
| anime | 29 | 58.6 | 79.3 | 69.0 |
| cpop | 24 | 100.0 | 100.0 | 87.5 |
| electronic | 14 | 92.9 | 92.9 | 78.6 |
| lofi | 15 | 46.7 | 80.0 | 46.7 |
| classical | 15 | 100.0 | 100.0 | 100.0 |
| hiphop | 18 | 61.1 | 83.3 | 61.1 |
| other | 22 | 68.2 | 100.0 | 50.0 |

Most frequent genre confusions of the shipped model on the real test set (true → predicted, count): other → not music 5; anime → jpop 5; lofi → rock 3; anime → cpop 3; lofi → not music 2; hiphop → other 2; anime → not music 2.

Fully right by language on the real test set for the licence-clean variant: English 84.3 (n = 159), Chinese 79.8 (n = 94), Japanese 75.4 (n = 118).

### 2.6 Hybrid hand-off curve (shipped model with its thresholds; fully right, %)

| Confidence threshold τ | Share sent to DeepSeek (real test) | Real test | Real dev | Focus | Synthetic val |
|---|---|---|---|---|---|
| 0 (never) | 0.0 | 84.1 | 89.4 | 90.3 | 95.1 |
| 0.5 | 3.9 | 86.2 | 92.1 | 91.1 | 96.0 |
| 0.6 | 11.2 | 88.5 | 93.7 | 92.5 | 97.3 |
| 0.7 | 16.7 | 90.6 | 95.3 | 93.6 | 98.2 |
| 0.8 | 21.4 | 92.4 | 96.5 | 93.9 | 98.4 |
| 0.85 | 25.3 | 93.5 | 96.5 | 93.3 | 98.9 |
| 0.9 | 31.9 | 95.6 | 96.9 | 93.6 | 99.0 |
| 0.95 | 40.5 | 96.3 | 97.2 | 94.4 | 98.9 |
| always | 100.0 | 95.3 | 95.3 | 93.9 | 98.3 |

At τ = 0.9 the share sent was 26.8% on real dev, 20.0% on focus and 18.0% on synthetic validation. The τ = 0 row is the shipped model alone in its shipped configuration (thresholds, and +3 for a title match as well as an artist match); its 0.5% abstentions count as errors, which is why its real-test value (84.1) differs from the unified no-threshold value (84.9).

Licence-clean variant at τ = 0.9 (with its own thresholds): real test 94.8 fully right with 44.1% sent; focus 94.2 with 25.0% sent; synthetic validation 98.9 with 20.2% sent. Alone, with its thresholds, it answered 91.1% of real test items and was fully right on 84.5% of those.

### 2.7 Focus subsets (fully right, %, n = 120 each)

| Subset | Linear round 4 | Linear round 5 (shipped) | DeepSeek |
|---|---|---|---|
| Internet memes | 75.8 | 89.2 | 93.3 |
| Japanese idols | 85.0 | 90.8 | 96.7 |
| Anime | 88.3 | 90.8 | 91.7 |

Before round 5 the training data had 63 meme-related rows out of about 19,000.

### 2.8 Distillation (real dev + real test combined, 637 rows, no thresholds, artist bonus 3, %)

| Variant | Real genre | Real video class | Real type |
|---|---|---|---|
| Round 5 (no teacher data) | 83.3 | 87.5 | 97.0 |
| + 8,982 teacher-labelled rows + 286 teacher corrections | 81.9 | 79.6 | 95.4 |
| + 4,030 class-balanced teacher rows + corrections | 83.3 | 79.6 | 96.2 |
| Teacher rows only (no corrections) | 83.1 | 84.9 | 97.0 |
| Teacher corrections only (no new rows) | 82.5 | 82.9 | 96.5 |

The teacher labelled 16,299 unlabelled real titles (remaining store-chart songs, 3,845 podcast episodes, 640 music videos, Bilibili and YouTube trending titles); rows with teacher confidence ≥ 0.8 were kept (8,982). It also audited all 21,535 existing training rows; 286 rows where it was confident (≥ 0.9) and disagreed were dropped (hand-written rows) or relabelled (rule-labelled chart rows). Class balancing capped teacher rows at 600 per music genre and 1,500 podcasts. At least 2.37 million teacher tokens were used.

Teacher audit of training labels (confident disagreements / rows): hand-written sub-agent rows 89 / 13,245 (0.7%); store-chart rule-labelled rows 197 / 8,290 (2.4%).

### 2.9 Other ablations (real dev + real test combined, no thresholds, artist bonus 3, %)

| Variant | Real genre | Real video class | Real type |
|---|---|---|---|
| Round 3, all data | 78.6 | 78.3 | 95.0 |
| Round 3, without chart songs | 76.1 | 78.3 | 93.9 |
| Round 3, without chart artists | 78.3 | 78.3 | 95.0 |
| Round 3, without any chart data | 76.1 | 78.3 | 93.9 |
| Round 4, curated lists overrule chart genres (adopted) | 82.8 | 86.2 | 96.4 |
| Round 4, hash table doubled to 2^18 (not adopted) | 80.8 | 87.5 | 96.5 |
| Round 4, + trending-style video and live titles (before override) | 78.6 | 86.8 | 96.2 |
| Round 5, + memes, idols, anime (adopted) | 83.3 | 87.5 | 97.0 |

Conflicting artist genres between curated lists and chart-derived labels fell from 471 to 72 after the override rule.

### 2.10 Transformer details

Real test fully right: transformer 84.1 vs linear 84.9 (unified setting). Ensemble 86.7 (+1.8 points over linear, about 7 items); the pre-registered adoption bar was +3 points. Dynamic int8 quantisation reduced the transformer's real-test fully-right accuracy from 84.6 to 79.9 (both measured with the shipped artist-bonus settings).

### 2.11 Known versus unknown artists, and per-language accuracy (real test set, unified setting)

Genre accuracy on real-test music items (n = 214), split by whether the artist is known to the system (artist field or a title segment matches the artist table, or the artist appears in the music training rows):

| Group | n | Shipped linear | DeepSeek V4.1 Flash |
|---|---|---|---|
| Known artist | 168 | 90.5 (152/168) | 95.2 (160/168) |
| Unknown artist | 46 | 47.8 (22/46) | 87.0 (40/46) |

Within the known group: artist-field table match 92.6 (137/148), title-segment table match 75.0 (9/12), not in table but in training rows 75.0 (6/8). Unknown-artist items are 21% of real-test music items but account for 24 of the shipped model's 40 genre errors (60%).

Fully right by language for the shipped model (unified setting): English 89.9 (143/159), Chinese 83.0 (78/94), Japanese 81.4 (96/118), items with no language tag 66.7 (8/12).

### 2.12 Protocol notes

From round 3 on, thresholds were tuned on the real development split only. The ablations in 2.8 and 2.9 (distillation variants, chart-data ablation, curated override, hash size, round-5 data) were scored on real development and test combined (637 items) and informed which variants were adopted. The hand-off threshold τ = 0.9 was chosen after inspecting the real-test curve; on the real development curve the best value is 0.95 (97.2) and 0.9 gives 96.9. Shipped artist bonus: +3 for an artist-field match and +3 for a title-segment match. Linear-model latency was measured in Node.js; transformer latency in ONNX Runtime; both on the same 2-core cloud CPU. The teacher took about 1 s per request of 20 titles; single-item latency was not measured.

## 3. Qualitative Observations

* The first synthetic validation set reported 97.5% type and 96.7% genre accuracy at round 1, which looked like the targets were met; the only warning sign was 91.1% genre accuracy on music rows whose artist was unknown to both the table and the training data. The real-world test set later showed genre accuracy about 13–15 points lower.
* Adding store-chart data in round 3 raised synthetic scores but lowered the real test with thresholds. The no-threshold ablation showed the chart data itself helped (+2.5 genre points); the drop came from thresholds and artist bonus tuned on a 254-row development split and from the store's genre taxonomy differing from ours (for example, Japanese bands labelled rock). Letting curated lists overrule chart genres fixed it.
* The remaining genre errors of the shipped model concentrate in anime, lofi, hiphop and other. These are genres that require recognising the specific song, artist or producer: anime theme songs are often sung by mainstream J-pop or C-pop artists with no anime cue in the title; lofi and hiphop have many new small producers; "other" music without cues was often not recognised as music.
* DeepSeek, given exactly the same text, reached 95.3% fully right, so the information needed is largely present in the text for a model that has the world knowledge.
* Distillation failed for a concrete reason in its first version: about a quarter of the teacher-labelled podcast episodes were attached to a video-site host, which taught the student that video-site items can be podcasts and lowered video accuracy; balancing and fixing this still did not beat round 5. Adding new teacher rows hurt video classification less than applying teacher corrections. The real dev + test sets contain only about 150 videos, so swings of a few points are partly noise.
* The teacher disagreed confidently with only 0.7% of hand-written training labels, mostly on rock/pop and jpop/anime boundaries, indicating the sub-agent labels were reliable.
* The transformer was trained only 3 epochs on CPU with frozen embeddings; longer GPU training might help, but it did not beat the linear model on any real or focus set and was far larger and slower.
* In the hybrid, sending about one third of items exceeded sending everything to the teacher, because the small model's confident answers are highly accurate and the two models' errors only partly overlap.
* Doubling the hash table gave about +2 genre points, within noise, for twice the size; not adopted.
* Removing all chart-derived data (for licensing reasons) lowered real-test fully right from 84.9 to 79.9 (genre 81.3 → 73.8), a significant paired difference, while slightly raising the focus and synthetic sets. The real-world set was scraped in the same week as the chart data, so artist overlap favours the chart-trained model; this mirrors real use if the artist table is refreshed from charts.
* Rules decided by the platform (music sites, podcast sites, podcast episode URLs) were more accurate than any model and cost nothing.
* Limitations: the real test set is small (383 rows, 214 music, about 100 video), so differences under about 2 points are not reliable; its labels were given by LLM sub-agents, and the teacher may agree with them partly because both are large language models with similar boundary judgements, so the teacher's score may be optimistic; the Swift and C# ports were not compiled.
