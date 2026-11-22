from pathlib import Path

import numpy as np
import tensorflow as tf

from main import load_data, prepare_data
from train_ann import build_model
from improve_ann import build_improved_model


OUTPUT_DIR = Path("output")

BASELINE_MODEL = OUTPUT_DIR / "cancer_ann.keras"
IMPROVED_MODEL = OUTPUT_DIR / "improved_cancer_ann.keras"


def test_dataset_structure():
    """Dataset should contain 600 patients and expected target."""

    df = load_data()

    assert len(df) == 600
    assert "diagnosis" in df.columns
    assert "patient_id" in df.columns

    # Diagnosis must contain only benign/malignant labels
    assert set(df["diagnosis"].unique()) == {0, 1}


def test_no_missing_values():
    """Cancer dataset should not contain missing values."""

    df = load_data()

    assert df.isna().sum().sum() == 0


def test_patient_id_removed_from_predictors():
    """Patient ID must never be used as an ANN predictor."""

    df = load_data()
    data = prepare_data(df)

    assert "patient_id" not in data["feature_names"]
    assert "diagnosis" not in data["feature_names"]

    # Expected 12 radiological predictors
    assert len(data["feature_names"]) == 12


def test_split_sizes_and_class_balance():
    """Verify 70/15/15 split and stratified class allocation."""

    df = load_data()
    data = prepare_data(df)

    assert data["X_train"].shape == (420, 12)
    assert data["X_val"].shape == (90, 12)
    assert data["X_test"].shape == (90, 12)

    # Confirm malignant cases remain represented
    assert np.sum(data["y_train"] == 1) == 196
    assert np.sum(data["y_val"] == 1) == 42
    assert np.sum(data["y_test"] == 1) == 42


def test_training_scaling():
    """
    StandardScaler should produce approximately
    zero means and unit standard deviations.
    """

    df = load_data()
    data = prepare_data(df)

    means = data["X_train"].mean(axis=0)
    stds = data["X_train"].std(axis=0)

    assert np.allclose(
        means,
        0.0,
        atol=1e-7
    )

    assert np.allclose(
        stds,
        1.0,
        atol=1e-7
    )


def test_baseline_ann_architecture():
    """Baseline ANN should accept 12 predictors and output one probability."""

    model = build_model(
        input_features=12
    )

    assert model.input_shape == (None, 12)
    assert model.output_shape == (None, 1)

    # Two hidden Dense layers + one output Dense layer
    dense_layers = [
        layer
        for layer in model.layers
        if isinstance(
            layer,
            tf.keras.layers.Dense
        )
    ]

    assert len(dense_layers) == 3
    assert dense_layers[-1].activation.__name__ == "sigmoid"


def test_improved_ann_architecture():
    """
    Improved ANN should include Dropout
    and one sigmoid output.
    """

    model = build_improved_model(
        input_features=12
    )

    assert model.input_shape == (None, 12)
    assert model.output_shape == (None, 1)

    dropout_layers = [
        layer
        for layer in model.layers
        if isinstance(
            layer,
            tf.keras.layers.Dropout
        )
    ]

    assert len(dropout_layers) == 2


def test_prediction_is_probability():
    """ANN output should always fall between 0 and 1."""

    model = build_model(
        input_features=12
    )

    sample = np.zeros(
        (1, 12),
        dtype=np.float32
    )

    probability = model(
        sample,
        training=False
    ).numpy()[0][0]

    assert 0.0 <= probability <= 1.0


def test_saved_models_exist_and_load():
    """Both trained Keras models should exist and load successfully."""

    assert BASELINE_MODEL.exists()
    assert IMPROVED_MODEL.exists()

    baseline = tf.keras.models.load_model(
        BASELINE_MODEL
    )

    improved = tf.keras.models.load_model(
        IMPROVED_MODEL
    )

    assert baseline.output_shape == (None, 1)
    assert improved.output_shape == (None, 1)