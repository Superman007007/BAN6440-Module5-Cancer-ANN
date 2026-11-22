from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from sklearn.utils.class_weight import compute_class_weight

import numpy as np

from main import load_data, prepare_data


OUTPUT_DIR = Path("output")
RANDOM_STATE = 42
THRESHOLD = 0.50

tf.keras.utils.set_random_seed(RANDOM_STATE)


def build_improved_model(input_features):
    """
    Improved ANN:
    - Two hidden layers
    - Dropout for generalization
    - Sigmoid output for binary cancer classification
    """

    model = tf.keras.Sequential([
        tf.keras.layers.Input(
            shape=(input_features,)
        ),

        tf.keras.layers.Dense(
            32,
            activation="relu"
        ),

        tf.keras.layers.Dropout(
            0.20
        ),

        tf.keras.layers.Dense(
            16,
            activation="relu"
        ),

        tf.keras.layers.Dropout(
            0.10
        ),

        tf.keras.layers.Dense(
            1,
            activation="sigmoid"
        )
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


def create_class_weights(y_train):
    """
    Compute balanced class weights and give the malignant
    class a small additional emphasis to reduce false negatives.
    """

    classes = np.unique(y_train)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train
    )

    class_weights = {
        int(cls): float(weight)
        for cls, weight in zip(
            classes,
            weights
        )
    }

    # Slightly emphasize malignant cases
    class_weights[1] *= 1.10

    return class_weights


def plot_history(history):
    """Save improved ANN training/validation curves."""

    # Accuracy chart
    plt.figure(figsize=(8, 5))

    plt.plot(
        history.history["accuracy"],
        label="Training Accuracy"
    )

    plt.plot(
        history.history["val_accuracy"],
        label="Validation Accuracy"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(
        "Improved ANN Training and Validation Accuracy"
    )

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR /
        "improved_training_accuracy.png",
        dpi=300
    )

    plt.close()

    # Loss chart
    plt.figure(figsize=(8, 5))

    plt.plot(
        history.history["loss"],
        label="Training Loss"
    )

    plt.plot(
        history.history["val_loss"],
        label="Validation Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Binary Cross-Entropy Loss")

    plt.title(
        "Improved ANN Training and Validation Loss"
    )

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR /
        "improved_training_loss.png",
        dpi=300
    )

    plt.close()


def save_confusion_matrix(cm):
    """Save confusion matrix for improved ANN."""

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Greens",
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
        "Improved ANN Cancer Diagnosis Confusion Matrix"
    )

    plt.xlabel("Predicted Diagnosis")
    plt.ylabel("Actual Diagnosis")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR /
        "improved_confusion_matrix.png",
        dpi=300
    )

    plt.close()


def main():

    OUTPUT_DIR.mkdir(exist_ok=True)

    # --------------------------------------------------
    # Load and prepare same dataset split
    # --------------------------------------------------

    print(
        "Loading and preprocessing cancer dataset..."
    )

    df = load_data()
    data = prepare_data(df)

    # --------------------------------------------------
    # Class weights
    # --------------------------------------------------

    class_weights = create_class_weights(
        data["y_train"]
    )

    print("\nClass weights:")
    print(class_weights)

    # --------------------------------------------------
    # Build improved ANN
    # --------------------------------------------------

    print("\nBuilding improved ANN...")

    model = build_improved_model(
        input_features=
        data["X_train"].shape[1]
    )

    model.summary()

    # --------------------------------------------------
    # Training callbacks
    # --------------------------------------------------

    early_stopping = (
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=20,
            restore_best_weights=True
        )
    )

    reduce_lr = (
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=7,
            min_lr=0.00001,
            verbose=1
        )
    )

    # --------------------------------------------------
    # Train
    # --------------------------------------------------

    print("\nTraining improved ANN...")

    history = model.fit(
        data["X_train"],
        data["y_train"],

        validation_data=(
            data["X_val"],
            data["y_val"]
        ),

        epochs=200,
        batch_size=16,

        class_weight=class_weights,

        callbacks=[
            early_stopping,
            reduce_lr
        ],

        verbose=1
    )

    # --------------------------------------------------
    # Training charts
    # --------------------------------------------------

    plot_history(history)

    # --------------------------------------------------
    # Test-set evaluation
    # --------------------------------------------------

    print(
        "\nEvaluating improved ANN "
        "on untouched test set..."
    )

    probabilities = model.predict(
        data["X_test"],
        verbose=0
    ).ravel()

    predictions = (
        probabilities >= THRESHOLD
    ).astype(int)

    actual = data["y_test"]

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

    # --------------------------------------------------
    # Results
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("IMPROVED ANN TEST PERFORMANCE")
    print("=" * 60)

    print(
        f"\nEpochs completed: "
        f"{len(history.history['loss'])}"
    )

    print(
        f"Best validation loss: "
        f"{min(history.history['val_loss']):.4f}"
    )

    print(
        f"\nAccuracy:  {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall:    {recall:.4f}"
    )

    print(
        f"F1 Score:  {f1:.4f}"
    )

    print(
        f"ROC-AUC:   {auc:.4f}"
    )

    print("\nConfusion Matrix:")
    print(cm)

    print(
        "\nTrue Negatives:  "
        f"{tn}"
    )

    print(
        "False Positives: "
        f"{fp}"
    )

    print(
        "False Negatives: "
        f"{fn}"
    )

    print(
        "True Positives:  "
        f"{tp}"
    )

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

    # --------------------------------------------------
    # Save model and chart
    # --------------------------------------------------

    model_file = (
        OUTPUT_DIR /
        "improved_cancer_ann.keras"
    )

    model.save(model_file)

    save_confusion_matrix(cm)

    print(
        f"\nSaved model: {model_file}"
    )

    print(
        "Saved chart: "
        "output/improved_training_accuracy.png"
    )

    print(
        "Saved chart: "
        "output/improved_training_loss.png"
    )

    print(
        "Saved chart: "
        "output/improved_confusion_matrix.png"
    )

    # --------------------------------------------------
    # Comparison to baseline
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("BASELINE REFERENCE")
    print("=" * 60)

    print("Baseline Accuracy:  0.9667")
    print("Baseline Precision: 0.9535")
    print("Baseline Recall:    0.9762")
    print("Baseline F1:        0.9647")
    print("Baseline ROC-AUC:   0.9965")
    print("Baseline FN:        1")

    print("\nImprovement priority:")
    print(
        "Reduce malignant false negatives while "
        "maintaining strong overall performance."
    )

    print(
        "\nIMPORTANT: These results are for an "
        "educational dataset and do not establish "
        "clinical validity."
    )


if __name__ == "__main__":
    main()