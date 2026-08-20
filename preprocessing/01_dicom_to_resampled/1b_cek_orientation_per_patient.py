import pandas as pd

df = pd.read_csv("dicom_audit.csv")

grouped = df.groupby("patient")["orientation"].nunique()

print("Patients with >1 orientation inside same patient:")
print(grouped[grouped > 1])
