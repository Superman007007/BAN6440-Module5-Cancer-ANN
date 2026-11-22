from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf

from main import load_data, prepare_data


OUTPUT_DIR = Path("output")
RANDOM_STATE = 42

# Make results more reproducible
tf.keras.utils.set_random_seed(RANDOM_STATE)


def build_model(input_features):
    """
    Build a compact ANN for binary cancer classification.

    Architecture:
    12 inputs -> 32 ReLU -> 16 ReLU -> 1 Sigmoid
    """

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_features,)),

        tf.keras.layers.Dense(
            32,
            activation="relu"
        ),

        tf.keras.layers.Dense(
            16,
            activation="relu"
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


def plot_training_history(history):
    """Save training and validation accuracy/loss curves."""

    OUTPUT_DIR.mkdir(exist_ok=True)

    # Accuracy
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
    plt.title("ANN Training and Validation Accuracy")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "training_accuracy.png",
        dpi=300
    )

    plt.close()

    # Loss
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
    plt.title("ANN Training and Validation Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "training_loss.png",
        dpi=300
    )

    plt.close()


def main():

    OUTPUT_DIR.mkdir(exist_ok=True)

    # -----------------------------------------
    # Load and preprocess data
    # -----------------------------------------

    print("Loading and preprocessing dataset...")

    df = load_data()
    data = prepare_data(df)

    # -----------------------------------------
    # Build ANN
    # -----------------------------------------

    print("\nBuilding ANN model...")

    model = build_model(
        input_features=data["X_train"].shape[1]
    )

    model.summary()

    # -----------------------------------------
    # Early stopping
    # -----------------------------------------

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=15,
        restore_best_weights=True
    )

    # -----------------------------------------
    # Train
    # -----------------------------------------

    print("\nTraining ANN...")

    history = model.fit(
        data["X_train"],
        data["y_train"],
        validation_data=(
            data["X_val"],
            data["y_val"]
        ),
        epochs=150,
        batch_size=16,
        callbacks=[early_stopping],
        verbose=1
    )

    # -----------------------------------------
    # Save model
    # -----------------------------------------

    model_path = OUTPUT_DIR / "cancer_ann.keras"

    model.save(model_path)

    # -----------------------------------------
    # Save training curves
    # -----------------------------------------

    plot_training_history(history)

    # -----------------------------------------
    # Validation performance
    # -----------------------------------------

    val_loss, val_accuracy = model.evaluate(
        data["X_val"],
        data["y_val"],
        verbose=0
    )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
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
        f"Final validation accuracy: "
        f"{val_accuracy:.4f}"
    )

    print(f"\nSaved model: {model_path}")

    print(
        "Saved chart: "
        "output/training_accuracy.png"
    )

    print(
        "Saved chart: "
        "output/training_loss.png"
    )

    print(
        "\nNOTE: The untouched test set has not "
        "been used for final evaluation yet."
    )


if __name__ == "__main__":
    main()