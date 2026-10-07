import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# Connect to MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Use the same experiment
mlflow.set_experiment("Salary Prediction")


# Load dataset
data = pd.read_csv("Salary_Data.csv")

# Input and target
X = data[["YearsExperience"]]
y = data["Salary"]


# Parameters
test_size = 0.2
random_state = 42
max_depth = 5


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=test_size,
    random_state=random_state
)


# Create Decision Tree model
model = DecisionTreeRegressor(
    max_depth=max_depth,
    random_state=random_state
)


# Start MLflow run
with mlflow.start_run(run_name="Decision Tree"):

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Metrics
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5

    # Log parameters
    mlflow.log_param("test_size", test_size)
    mlflow.log_param("random_state", random_state)
    mlflow.log_param("max_depth", max_depth)

    # Log metrics
    mlflow.log_metric("R2_Score", r2)
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("RMSE", rmse)

    # Log model
    mlflow.sklearn.log_model(
        model,
        name="decision_tree_model"
    )

    print("===== DECISION TREE =====")
    print("Max Depth :", max_depth)
    print("R2 Score :", r2)
    print("Performance :", r2 * 100, "%")
    print("MAE :", mae)
    print("RMSE :", rmse)
    print("Decision Tree tracked successfully in MLflow!")
