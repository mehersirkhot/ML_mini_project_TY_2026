
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from utils.preprocess import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    MODEL_PATH,
    clean_dataset,
)


def prepare_features(df, feature_columns):
    """
    Convert every feature into a consistent text format.
    Missing values remain missing and are imputed later.
    """
    result = df[feature_columns].copy()

    for column in feature_columns:
        result[column] = result[column].apply(
            lambda value: str(value).strip()
            if pd.notna(value)
            else np.nan
        )

    return result


def build_pipeline(model, feature_columns):
    """
    Build preprocessing and classification into one pipeline.
    """

    preprocessing = ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(strategy="most_frequent"),
                        ),
                        (
                            "encoder",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                feature_columns,
            )
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            ("classifier", model),
        ]
    )


def train_and_select_model(df, model_path=MODEL_PATH):
    """
    Train four classification algorithms and select
    the best model based on test accuracy.
    """

    df = clean_dataset(df)

    available_features = [
        column
        for column in FEATURE_COLUMNS
        if column in df.columns
    ]

    if not available_features:
        raise ValueError(
            "No valid input features were found in the dataset."
        )

    df = df.dropna(subset=[TARGET_COLUMN]).copy()

    if len(df) < 12:
        raise ValueError(
            "At least 12 valid survey responses are required."
        )

    X = prepare_features(df, available_features)

    y = df[TARGET_COLUMN].astype(str).str.strip()

    if y.nunique() < 2:
        raise ValueError(
            "At least two different MHT-CET score categories are required."
        )

    class_counts = y.value_counts()

    if class_counts.min() < 2:
        raise ValueError(
            "Each score category must have at least two responses."
        )

    # Convert target categories into numerical labels.
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Split data while preserving category proportions.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.25,
        random_state=42,
        stratify=y_encoded,
    )

    # Define the four classification algorithms.
    models = {
        "Naive Bayes": GaussianNB(),

        "Decision Tree": DecisionTreeClassifier(
            max_depth=5,
            random_state=42,
            class_weight="balanced",
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            random_state=42,
            class_weight="balanced",
        ),

        "K-Nearest Neighbors": KNeighborsClassifier(
            n_neighbors=min(5, len(X_train)),
        ),
    }

    results = []

    best_model = None
    best_name = None
    best_accuracy = -1
    best_predictions = None

    # Train and evaluate each algorithm.
    for name, algorithm in models.items():

        pipeline = build_pipeline(
            algorithm,
            available_features,
        )

        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)

        accuracy = accuracy_score(
            y_test,
            predictions,
        )

        precision = precision_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0,
        )

        recall = recall_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0,
        )

        f1 = f1_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0,
        )

        results.append(
            {
                "Model": name,
                "Accuracy (%)": round(accuracy * 100, 2),
                "Precision (%)": round(precision * 100, 2),
                "Recall (%)": round(recall * 100, 2),
                "F1 Score (%)": round(f1 * 100, 2),
            }
        )

        # Keep the model with the highest test accuracy.
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = pipeline
            best_name = name
            best_predictions = predictions

    # Generate evaluation reports for the selected model.
    class_labels = np.arange(len(label_encoder.classes_))

    matrix = confusion_matrix(
        y_test,
        best_predictions,
        labels=class_labels,
    )

    report = classification_report(
        y_test,
        best_predictions,
        labels=class_labels,
        target_names=label_encoder.classes_,
        zero_division=0,
        output_dict=True,
    )

    payload = {
        "model": best_model,
        "model_name": best_name,
        "feature_columns": available_features,
        "target_column": TARGET_COLUMN,
        "label_encoder": label_encoder,
        "class_names": list(label_encoder.classes_),
        "results": results,
        "confusion_matrix": matrix.tolist(),
        "classification_report": report,
        "training_rows": len(X_train),
        "testing_rows": len(X_test),
    }

    # Save the newly trained model.
    model_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(model_path, "wb") as file:
        pickle.dump(payload, file)

    return payload


def load_model(model_path=MODEL_PATH):
    """
    Load a previously trained model.
    """

    if not model_path.exists():
        return None

    try:
        with open(model_path, "rb") as file:
            return pickle.load(file)

    except Exception:
        return None


def predict_student(input_data, model_payload=None):
    """
    Predict the MHT-CET score category for a new response.
    """

    payload = model_payload or load_model()

    if payload is None:
        raise ValueError(
            "No trained model found. Train the models first."
        )

    model = payload["model"]

    label_encoder = payload["label_encoder"]

    feature_columns = payload["feature_columns"]

    # Create a one-row input DataFrame.
    input_df = pd.DataFrame([input_data])

    # Add any missing features.
    for column in feature_columns:
        if column not in input_df.columns:
            input_df[column] = np.nan

    # Prepare features in the same way as training data.
    input_df = prepare_features(
        input_df,
        feature_columns,
    )

    # Predict the encoded category.
    prediction = model.predict(input_df)[0]

    predicted_category = label_encoder.inverse_transform(
        [prediction]
    )[0]

    # Calculate probabilities for each category.
    probabilities = {}

    if hasattr(model, "predict_proba"):

        values = model.predict_proba(input_df)[0]

        classifier_classes = (
            model.named_steps["classifier"].classes_
        )

        for class_index, probability in zip(
            classifier_classes,
            values,
        ):

            category = label_encoder.inverse_transform(
                [int(class_index)]
            )[0]

            probabilities[category] = round(
                float(probability) * 100,
                2,
            )

    return {
        "predicted_category": predicted_category,
        "probabilities": probabilities,
    }
