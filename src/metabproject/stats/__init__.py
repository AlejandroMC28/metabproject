"""Statistical analysis: per-feature contrasts and factorial comparisons."""

from metabproject.stats.anova import anova_1way, anova_2way, welch_anova
from metabproject.stats.comparison_runner import ComparisonResult, run_comparison
from metabproject.stats.comparisons import Comparison, load_comparisons
from metabproject.stats.contrasts import Contrast, load_contrasts, resolve_samples
from metabproject.stats.diagnostics import levene_per_feature, shapiro_per_feature
from metabproject.stats.kruskal import kruskal_wallis
from metabproject.stats.posthoc import dunn, games_howell, tukey_hsd
from metabproject.stats.univariate import welch_ttest

__all__ = [
    "Comparison",
    "ComparisonResult",
    "Contrast",
    "anova_1way",
    "anova_2way",
    "dunn",
    "games_howell",
    "kruskal_wallis",
    "levene_per_feature",
    "load_comparisons",
    "load_contrasts",
    "resolve_samples",
    "run_comparison",
    "shapiro_per_feature",
    "tukey_hsd",
    "welch_anova",
    "welch_ttest",
]
