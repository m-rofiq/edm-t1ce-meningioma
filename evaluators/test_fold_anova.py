import numpy as np
from fold_anova import fold_anova

fold_results = {
    0: np.random.normal(30, 1, 10),
    1: np.random.normal(30, 1, 10),
    2: np.random.normal(30, 1, 10),
    3: np.random.normal(30, 1, 10),
    4: np.random.normal(30, 1, 10),
}

F = fold_anova(fold_results)

print("F-statistic:", F)