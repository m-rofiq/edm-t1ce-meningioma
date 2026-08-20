import numpy as np

def bootstrap_ci(values, n_bootstrap=1000, ci=95, seed=42):

    rng = np.random.default_rng(seed)

    values = np.asarray(values, dtype=np.float64)

    boot_means = []

    for _ in range(n_bootstrap):

        sample = rng.choice(values, size=len(values), replace=True)

        boot_means.append(np.mean(sample))

    lower = np.percentile(boot_means, (100 - ci) / 2)
    upper = np.percentile(boot_means, 100 - (100 - ci) / 2)

    return np.mean(values), lower, upper