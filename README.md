[![DOI](https://zenodo.org/badge/1274837168.svg)](https://doi.org/10.5281/zenodo.20768148)

## Reproducing the paper

`code/cdw_lda.py` reproduces the four-topic LDA model reported in the paper (136 x 1,671
document-term matrix; 30/12/69/25 documents per topic) and the robustness analyses
(multi-seed perplexity and C_V coherence, Friedman and Wilcoxon signed-rank tests,
convergence and topic stability).

```
pip install -r code/requirements.txt
python code/cdw_lda.py data/136_Full.csv outputs
```

`outputs/` contains the results of this script (tables, statistics, and figures used in the paper).
`code/STM.ipynb` is the earlier exploratory notebook and does not reproduce the reported model.
