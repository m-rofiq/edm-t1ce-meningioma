import numpy as np

def fold_anova(fold_results_dict):
    """
    Manual one-way ANOVA (no scipy dependency)
    fold_results_dict = {
        0: [values],
        1: [values],
        ...
    }
    """

    groups = [np.array(fold_results_dict[k]) for k in fold_results_dict]
    k = len(groups)

    all_values = np.concatenate(groups)
    N = len(all_values)

    grand_mean = np.mean(all_values)

    # Between-group variance
    ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups)

    # Within-group variance
    ss_within = sum(sum((g - np.mean(g))**2) for g in groups)

    df_between = k - 1
    df_within = N - k

    ms_between = ss_between / df_between
    ms_within = ss_within / df_within

    F_stat = ms_between / (ms_within + 1e-8)

    return F_stat