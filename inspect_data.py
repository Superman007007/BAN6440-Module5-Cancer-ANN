from pathlib import Path
import pandas as pd


# ---------------------------------------------------------
# File location
# ---------------------------------------------------------

DATA_FILE = Path(
    "data/BAN6440 M5 mdanderson_cancer_diagnosis.csv"
)


# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_FILE.resolve()}"
    )

df = pd.read_csv(DATA_FILE)


# ---------------------------------------------------------
# Basic dataset information
# ---------------------------------------------------------

print("=" * 65)
print("MD ANDERSON CANCER DIAGNOSIS DATASET INSPECTION")
print("=" * 65)

print(f"\nDataset path:\n{DATA_FILE.resolve()}")

print(f"\nRows: {len(df)}")
print(f"Columns: {len(df.columns)}")


# ---------------------------------------------------------
# Column names
# ---------------------------------------------------------

print("\n" + "=" * 65)
print("COLUMN NAMES")
print("=" * 65)

for index, column in enumerate(df.columns, start=1):
    print(f"{index}. {column}")


# ---------------------------------------------------------
# First five records
# ---------------------------------------------------------

print("\n" + "=" * 65)
print("FIRST FIVE RECORDS")
print("=" * 65)

print(df.head().to_string())


# ---------------------------------------------------------
# Data types
# ---------------------------------------------------------

print("\n" + "=" * 65)
print("DATA TYPES")
print("=" * 65)

print(df.dtypes)


# ---------------------------------------------------------
# Missing values
# ---------------------------------------------------------

print("\n" + "=" * 65)
print("MISSING VALUES")
print("=" * 65)

missing_values = df.isna().sum()

print(missing_values)

print(
    f"\nTotal missing values: "
    f"{missing_values.sum()}"
)


# ---------------------------------------------------------
# Duplicate records
# ---------------------------------------------------------

print("\n" + "=" * 65)
print("DUPLICATE RECORDS")
print("=" * 65)

duplicate_count = df.duplicated().sum()

print(
    f"Duplicate rows: {duplicate_count}"
)


# ---------------------------------------------------------
# Diagnosis distribution
# ---------------------------------------------------------

print("\n" + "=" * 65)
print("DIAGNOSIS DISTRIBUTION")
print("=" * 65)

if "diagnosis" in df.columns:

    diagnosis_counts = (
        df["diagnosis"]
        .value_counts()
        .sort_index()
    )

    print("\nCounts:")
    print(diagnosis_counts)

    diagnosis_percentages = (
        df["diagnosis"]
        .value_counts(normalize=True)
        .sort_index()
        .mul(100)
        .round(2)
    )

    print("\nPercentages:")
    print(diagnosis_percentages)

    print(
        "\nDiagnosis meaning:"
        "\n0 = Benign"
        "\n1 = Malignant"
    )

else:
    print(
        "WARNING: 'diagnosis' column was not found."
    )


# ---------------------------------------------------------
# Check unique values
# ---------------------------------------------------------

if "diagnosis" in df.columns:

    print("\n" + "=" * 65)
    print("DIAGNOSIS UNIQUE VALUES")
    print("=" * 65)

    print(
        sorted(
            df["diagnosis"]
            .dropna()
            .unique()
            .tolist()
        )
    )


# ---------------------------------------------------------
# Numeric feature summary
# ---------------------------------------------------------

print("\n" + "=" * 65)
print("NUMERIC FEATURE SUMMARY")
print("=" * 65)

numeric_summary = (
    df.describe()
    .T
    .round(3)
)

print(
    numeric_summary.to_string()
)


# ---------------------------------------------------------
# Potential identifier columns
# ---------------------------------------------------------

print("\n" + "=" * 65)
print("POTENTIAL IDENTIFIER COLUMNS")
print("=" * 65)

possible_id_columns = [
    column
    for column in df.columns
    if "id" in column.lower()
]

if possible_id_columns:

    for column in possible_id_columns:

        unique_count = (
            df[column]
            .nunique(dropna=False)
        )

        print(
            f"{column}: "
            f"{unique_count} unique values"
        )

else:
    print("No obvious ID columns found.")


# ---------------------------------------------------------
# Feature correlations with diagnosis
# ---------------------------------------------------------

if "diagnosis" in df.columns:

    numeric_df = df.select_dtypes(
        include="number"
    )

    if "diagnosis" in numeric_df.columns:

        print("\n" + "=" * 65)
        print("CORRELATION WITH DIAGNOSIS")
        print("=" * 65)

        diagnosis_correlation = (
            numeric_df
            .corr()["diagnosis"]
            .drop("diagnosis")
            .sort_values(
                key=lambda x: x.abs(),
                ascending=False
            )
        )

        print(
            diagnosis_correlation
            .round(3)
            .to_string()
        )


# ---------------------------------------------------------
# Final inspection result
# ---------------------------------------------------------

print("\n" + "=" * 65)
print("DATASET INSPECTION COMPLETE")
print("=" * 65)

print(
    "\nNext step:"
    "\nUse these results to decide which columns should be "
    "removed, scaled, and used to train the TensorFlow ANN."
)