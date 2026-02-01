"""MLflow Demo Script.

This script demonstrates how to train a Logistic Regression model on the Iris dataset,
track the experiment using MLflow, and log metrics, parameters, and the model itself.
"""

from typing import cast

import mlflow
import pandas as pd
from mlflow import sklearn as mlflow_sklearn
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.utils import Bunch


def main() -> None:
    """Train a simple Iris classifier and log it to MLflow."""
    # Set the MLflow tracking URI to point to the local MLflow server
    try:
        mlflow.set_tracking_uri(uri="http://localhost:5000")
        # Define the experiment name for organization in MLflow
        mlflow.set_experiment("MLflow Tutorial")
    except MlflowException as exc:
        print(f"Warning: Could not connect to MLflow server: {exc}")
        print("Falling back to local file tracking.")
        # Fallback to local file-based tracking
        mlflow.set_tracking_uri(uri="./mlruns")
        mlflow.set_experiment("MLflow Tutorial")

    # Load the Iris dataset from scikit-learn
    # Some type checkers incorrectly infer this as a tuple, so handle both cases.
    iris_raw = datasets.load_iris()
    iris_bunch = iris_raw[0] if isinstance(iris_raw, tuple) else iris_raw
    iris = cast(Bunch, iris_bunch)
    x = iris.data
    y = iris.target
    feature_names = iris.feature_names

    # Split the data into training (80%) and testing (20%) sets
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    # Define hyperparameters for the Logistic Regression model
    params = {
        # The optimization algorithm used to fit logistic regression
        "solver": "lbfgs",
        # Maximum number of training iterations allowed for the solver.
        "max_iter": 1000,
        # Seed for the model’s internal randomness (used in some solver
        # paths / data shuffling depending on configuration).
        # Setting it makes results more reproducible run-to-run.
        "random_state": 8888,
    }

    # Optional: enable MLflow autologging for scikit-learn.
    # This must be enabled BEFORE training (before model.fit).
    # If you enable autologging, you may want to remove the manual mlflow.log_* calls below.
    # mlflow_sklearn.autolog()

    # Creates an untrained/unfitted estimator (model object) instance
    model = LogisticRegression(**params)
    # Fit/train the Logistic Regression estimator on the training data
    model.fit(x_train, y_train)

    # Initialize model_info explicitly
    model_info = None

    # Predict on the test set to evaluate performanc
    y_pred = model.predict(x_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    # Use 'weighted' average for multiclass classification to handle label imbalance
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    # Start an MLflow run to log information
    with mlflow.start_run():
        # Log the hyperparameters used for training
        mlflow.log_params(params)

        # Log the calculated evaluation metrics
        # Note: Metrics are cast to float to ensure compatibility with MLflow logging
        mlflow.log_metric("accuracy", float(accuracy))
        mlflow.log_metric("precision", float(precision))
        mlflow.log_metric("recall", float(recall))
        mlflow.log_metric("f1", float(f1))

        # Set a tag to describe the run
        mlflow.set_tag("Training Info", "Basic LR model for iris data")

        # Infer the model signature (input and output schema)
        signature = infer_signature(x_train, model.predict(x_train))

        # Log the model artifact to MLflow
        model_info = mlflow_sklearn.log_model(
            sk_model=model,
            artifact_path="iris-model",
            signature=signature,
            input_example=x_train,
            registered_model_name="iris-model-demo",
        )

    if model_info:
        loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
        predictions = loaded_model.predict(x_test)
        print("Model loaded and prediction successful.")

        result = pd.DataFrame(x_test, columns=feature_names)
        result["actual_class"] = y_test
        result["predicted_class"] = predictions

        print(result[:4])


if __name__ == "__main__":
    main()
