from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DATA_FILE = Path(
    "data/BAN6440 M5 mdanderson_cancer_diagnosis.csv"
)

OUTPUT_DIR = Path("output")

TARGET = "diagnosis"
ID_COLUMN = "patient_id"

RANDOM_STATE = 42


def load_data():
    """Load and validate the cancer diagnosis dataset."""

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE.resolve()}"
        )

    df = pd.read_csv(DATA_FILE)

    if TARGET not in df.columns:
        raise ValueError(
            f"Required target column '{TARGET}' not found."
        )

    if df.isna().sum().sum() > 0:
        raise ValueError(
            "Dataset contains missing values."
        )

    return df


def prepare_data(df):
    """
    Remove the patient identifier, split the dataset,
    and scale predictors without data leakage.
    """

    # patient_id is only an identifier
    X = df.drop(
        columns=[ID_COLUMN, TARGET]
    )

    y = df[TARGET]

    feature_names = X.columns.tolist()

    # 70% training, 30% temporary
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        stratify=y,
        random_state=RANDOM_STATE
    )

    # Split temporary data equally:
    # 15% validation and 15% testing
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=RANDOM_STATE
    )

    # Fit scaler ONLY to training data
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return {
        "feature_names": feature_names,
        "X_train": X_train_scaled,
        "X_val": X_val_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train.to_numpy(),
        "y_val": y_val.to_numpy(),
        "y_test": y_test.to_numpy(),
        "scaler": scaler
    }


def print_split_summary(data):
    """Display the prepared dataset dimensions and class balance."""

    print("=" * 60)
    print("PREPROCESSING RESULTS")
    print("=" * 60)

    print(
        f"\nPredictor features: "
        f"{len(data['feature_names'])}"
    )

    print("\nFeatures:")

    for feature in data["feature_names"]:
        print(f"- {feature}")

    print("\nDataset splits:")

    print(
        f"Training:   "
        f"{data['X_train'].shape}"
    )

    print(
        f"Validation: "
        f"{data['X_val'].shape}"
    )

    print(
        f"Testing:    "
        f"{data['X_test'].shape}"
    )

    print("\nClass balance:")

    for name in [
        "y_train",
        "y_val",
        "y_test"
    ]:

        values = data[name]

        benign = (values == 0).sum()
        malignant = (values == 1).sum()

        print(
            f"{name}: "
            f"Benign={benign}, "
            f"Malignant={malignant}"
        )

    print("\nScaling check:")

    print(
        "Training feature means "
        "(approximately zero):"
    )

    print(
        data["X_train"].mean(axis=0)
    )

    print(
        "\nTraining feature standard deviations "
        "(approximately one):"
    )

    print(
        data["X_train"].std(axis=0)
    )


def main():

    OUTPUT_DIR.mkdir(exist_ok=True)

    print("Loading cancer diagnosis dataset...")

    df = load_data()

    print("Preparing dataset...")

    prepared_data = prepare_data(df)

    print_split_summary(prepared_data)

    print(
        "\nPreprocessing completed successfully."
    )


if __name__ == "__main__":
    main()