import pandas as pd
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Salary Prediction")

data = pd.read_csv("Salary_Data.csv")

X = data[["YearsExperience"]]
y = data["Salary"]

test_size = 0.2
random_state = 42
n_estimators = 100
max_depth = 5

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=test_size,
    random_state=random_state
)

model = RandomForestRegressor(
    n_estimators=n_estimators,
    max_depth=max_depth,
    random_state=random_state
)

with mlflow.start_run(run_name="Random Forest"):

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5

    mlflow.log_params({
        "test_size": test_size,
        "random_state": random_state,
        "n_estimators": n_estimators,
        "max_depth": max_depth
    })

    mlflow.log_metrics({
        "R2_Score": r2,
        "MAE": mae,
        "RMSE": rmse
    })

    plt.figure(figsize=(8, 5))
    plt.scatter(y_test, y_pred)
    plt.xlabel("Actual Salary")
    plt.ylabel("Predicted Salary")
    plt.title("Random Forest - Actual vs Predicted")
    mlflow.log_figure(plt.gcf(), "random_forest_plot.png")
    plt.close()

    results = pd.DataFrame({
        "ActualSalary": y_test.values,
        "PredictedSalary": y_pred
    })
    results.to_csv("random_forest_predictions.csv", index=False)
    mlflow.log_artifact("random_forest_predictions.csv")

    mlflow.sklearn.log_model(
        model,
        name="random_forest_model"
    )

    print("===== RANDOM FOREST =====")
    print("R2 Score:", r2)
    print("Performance:", r2 * 100, "%")
    print("MAE:", mae)
    print("RMSE:", rmse)
    print("Random Forest tracked successfully!")
