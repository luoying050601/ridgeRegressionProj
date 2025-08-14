import numpy as np
import statsmodels.stats.multitest as sm
# bonferroni : one-step correction
# sidak : one-step correction
# holm-sidak : step down method using Sidak adjustments
# holm : step-down method using Bonferroni adjustments
# simes-hochberg : step-up method (independent)
# hommel : closed method based on Simes tests (non-negative)
# fdr_bh : Benjamini/Hochberg (non-negative)
# fdr_by : Benjamini/Yekutieli (negative)
# fdr_tsbh : two stage fdr correction (non-negative)
# fdr_tsbky : two stage fdr correction (non-negative)
sm.multipletests(pvals, alpha=0.05, method='fdr_bh', is_sorted=False, returnsorted=False)