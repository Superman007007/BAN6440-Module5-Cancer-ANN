from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

from main import load_data, prepare_data


OUTPUT_DIR = Path("output")
MODEL_FILE = OUTPUT_DIR / "cancer_ann.keras"

THRESHOLD = 0.50


def evaluate_model():

    # -----------------------------------------------------
    # Load and preprocess data using the same procedure
    # -----------------------------------------------------

    print("Loading and preprocessing dataset...")

    df = load_data()
    data = prepare_data(df)

    # -----------------------------------------------------
    # Load trained ANN
    # -----------------------------------------------------

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_FILE}"
        )

    print("Loading trained ANN...")

    model = tf.keras.models.load_model(
        MODEL_FILE
    )

    # -----------------------------------------------------
    # Generate malignant probabilities
    # -----------------------------------------------------

    print("Predicting untouched test patients...")

    probabilities = model.predict(
        data["X_test"],
        verbose=0
    ).ravel()

    predictions = (
        probabilities >= THRESHOLD
    ).astype(int)

    actual = data["y_test"]

    # -----------------------------------------------------
    # Performance metrics
    # -----------------------------------------------------

    accuracy = accuracy_score(
        actual,
        predictions
    )

    precision = precision_score(
        actual,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        actual,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        actual,
        predictions,
        zero_division=0
    )

    auc = roc_auc_score(
        actual,
        probabilities
    )

    cm = confusion_matrix(
        actual,
        predictions
    )

    tn, fp, fn, tp = cm.ravel()

    # -----------------------------------------------------
    # Display results
    # -----------------------------------------------------

    print("\n" + "=" * 60)
    print("FINAL TEST-SET PERFORMANCE")
    print("=" * 60)

    print(f"\nTest patients: {len(actual)}")
    print(f"Decision threshold: {THRESHOLD:.2f}")

    print(f"\nAccuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nConfusion Matrix Interpretation:")
    print(f"True Negatives  (Benign correctly identified): {tn}")
    print(f"False Positives (Benign predicted malignant):  {fp}")
    print(f"False Negatives (Malignant predicted benign):  {fn}")
    print(f"True Positives  (Malignant correctly identified): {tp}")

    print("\nClassification Report:")

    print(
        classification_report(
            actual,
            predictions,
            target_names=[
                "Benign",
                "Malignant"
            ],
            digits=4,
            zero_division=0
        )
    )

    # -----------------------------------------------------
    # Save confusion matrix chart
    # -----------------------------------------------------

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[
            "Predicted Benign",
            "Predicted Malignant"
        ],
        yticklabels=[
            "Actual Benign",
            "Actual Malignant"
        ]
    )

    plt.title(
        "ANN Cancer Diagnosis Confusion Matrix"
    )

    plt.ylabel("Actual Diagnosis")
    plt.xlabel("Predicted Diagnosis")

    plt.tight_layout()

    matrix_file = (
        OUTPUT_DIR /
        "confusion_matrix.png"
    )

    plt.savefig(
        matrix_file,
        dpi=300
    )

    plt.close()

    # -----------------------------------------------------
    # Save detailed predictions
    # -----------------------------------------------------

    results = df.iloc[
        data["y_test"].shape[0] * 0:
        data["y_test"].shape[0]
    ].copy()

    # Instead of relying on original row ordering,
    # save evaluation arrays directly.
    import pandas as pd

    prediction_results = pd.DataFrame({
        "actual_diagnosis": actual,
        "malignant_probability": probabilities,
        "predicted_diagnosis": predictions
    })

    prediction_file = (
        OUTPUT_DIR /
        "test_predictions.csv"
    )

    prediction_results.to_csv(
        prediction_file,
        index=False
    )

    # -----------------------------------------------------
    # Final safety-oriented interpretation
    # -----------------------------------------------------

    print(f"\nSaved: {matrix_file}")
    print(f"Saved: {prediction_file}")

    print("\nClinical evaluation priority:")

    if fn == 0:
        print(
            "No malignant cases were missed "
            "in this test sample."
        )
    else:
        print(
            f"{fn} malignant patient(s) were "
            "incorrectly classified as benign."
        )

    print(
        "\nIMPORTANT: Performance on this educational "
        "dataset does not establish clinical validity. "
        "External validation and clinician oversight "
        "would be required before real-world use."
    )


if __name__ == "__main__":
    evaluate_model()