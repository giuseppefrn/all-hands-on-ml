
# Hyperparameter Tuning with GridSearchCV, Optuna, and MLflow

## Import libraries
import numpy as np
import mlflow
import mlflow.sklearn
import optuna
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

## Load data
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

## Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ----------------------------------------
# GridSearchCV Example
# ----------------------------------------
print("\n=== GridSearchCV Example ===")
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [3, 5, None]
}
grid_model = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3)
grid_model.fit(X_train, y_train)
print("Best parameters from GridSearchCV:", grid_model.best_params_)

# ----------------------------------------
# Optuna Tuning with MLflow Tracking
# ----------------------------------------
print("\n=== Optuna + MLflow Example ===")

def objective(trial):
    n_estimators = trial.suggest_int("n_estimators", 50, 150)
    max_depth = trial.suggest_int("max_depth", 2, 10)
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)

    with mlflow.start_run(nested=True):
        mlflow.log_params({"n_estimators": n_estimators, "max_depth": max_depth})
        scores = cross_val_score(clf, X_train, y_train, cv=3, scoring='accuracy')
        mean_score = np.mean(scores)
        mlflow.log_metric("cv_accuracy", mean_score)
        return 1 - mean_score  # Minimize 1 - accuracy

mlflow.set_experiment("optuna_rf_tuning")
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=10)

print("Best trial:")
print(study.best_trial.params)

# ----------------------------------------
# Final Model Evaluation with MLflow Tracking and Model Registry
# ----------------------------------------

# Extract the best hyperparameters from Optuna study
best_params = study.best_trial.params

# Create and train the final model with the best parameters
final_model = RandomForestClassifier(**best_params, random_state=42)
final_model.fit(X_train, y_train)

# Make predictions
y_pred = final_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("\nFinal Model Accuracy on Test Set:", accuracy)

# Log final model and metrics to MLflow and register the model
with mlflow.start_run(run_name="final_model") as run:
    # Log best hyperparameters
    mlflow.log_params(best_params)

    # Log final accuracy
    mlflow.log_metric("test_accuracy", accuracy)

    # Log and register the model
    mlflow.sklearn.log_model(
        sk_model=final_model,
        artifact_path="final_rf_model",
        registered_model_name="Final_RF_Model"  # The model will be registered in the MLflow Model Registry
    )

    print("\nFinal model logged and registered in MLflow. Run ID:", run.info.run_id)
