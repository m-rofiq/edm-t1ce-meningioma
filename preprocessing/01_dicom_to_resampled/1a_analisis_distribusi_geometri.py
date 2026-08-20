import pandas as pd

df = pd.read_csv("dicom_audit.csv")

print("\n=== ORIENTATION DISTRIBUTION ===")
print(df["orientation"].value_counts().head(10))

print("\n=== PIXEL SPACING DISTRIBUTION ===")
print(df["pixel_spacing"].value_counts())

print("\n=== SLICE THICKNESS DISTRIBUTION ===")
print(df["slice_thickness"].value_counts())
