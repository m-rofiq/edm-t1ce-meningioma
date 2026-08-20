import numpy as np
from bootstrap_ci import bootstrap_ci

# dummy patient-level metrics
patient_psnr = np.random.normal(30, 2, size=12)

mean, lower, upper = bootstrap_ci(patient_psnr)

print("Mean:", mean)
print("95% CI:", lower, "-", upper)