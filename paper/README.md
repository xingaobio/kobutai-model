# Paper: Knowledge or Capacity? Lessons from a 2 MB On-Device Classifier for What Is Playing

Xin Gao, 2026. Preprint.

- **PDF:** [`kobutai-paper.pdf`](kobutai-paper.pdf)
- **arXiv source:** [`arxiv-source.zip`](arxiv-source.zip) (`paper.tex`, `paper.bbl`, `refs.bib`, figures; compiles with pdfLaTeX)
- **Source and figures:** [`source/`](source) (`make_figures.py` redraws every figure from the numbers in the experimental log)

## How it was written

The paper was drafted with the [PaperOrchestra](https://github.com/Ar9av/PaperOrchestra) skill pack (an implementation of Song et al., *PaperOrchestra: A Multi-Agent Framework for Automated AI Research Paper Writing*, arXiv:2604.05018) running in Claude Code: outline → figures and literature review in parallel → section writing → simulated peer review and revision. Everything it used is in [`process/`](process):

| File | What |
|---|---|
| `idea.md`, `experimental_log.md` | The inputs: method summary and every number used in the paper, taken from the research report and results in this repository |
| `outline.json`, `research_brief.md` | Paper plan and literature context |
| `citation_pool.json` | The 42 references, each confirmed against Semantic Scholar, arXiv or Crossref records |
| `reviews/` | Two rounds of independent simulated review and line-by-line fact check (overall 63.5 → 68.7; all numbers verified) |
| `provenance.json` | Hashes of inputs and outputs |

Section 2.11 of the experimental log (known vs unknown artists) and the confidence intervals and McNemar tests were computed for the paper from the models and cached answers in this repository.

## Licence

The paper text and figures: CC BY 4.0. Code: MIT.
