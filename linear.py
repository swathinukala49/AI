import pandas as pd
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# MLflow connection
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Salary Prediction")


data = pd.read_csv("Salary_Data.csv")

X = data[["YearsExperience"]]
y = data["Salary"]


test_size = 0.2
random_state = 42


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=test_size,
    random_state=random_state
)


model = LinearRegression()


with mlflow.start_run(run_name="Linear Regression"):

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5

    mlflow.log_params({
        "test_size": test_size,
        "random_state": random_state
    })

    mlflow.log_metrics({
        "R2_Score": r2,
        "MAE": mae,
        "RMSE": rmse
    })

    plt.figure(figsize=(8, 5))
    plt.scatter(X_test, y_test, label="Actual")
    plt.plot(X_test, y_pred, label="Predicted")
    plt.xlabel("Years of Experience")
    plt.ylabel("Salary")
    plt.title("Linear Regression - Actual vs Predicted")
    plt.legend()

    mlflow.log_figure(plt.gcf(), "linear_regression_plot.png")
    plt.close()

    results = pd.DataFrame({
        "YearsExperience": X_test["YearsExperience"].values,
        "ActualSalary": y_test.values,
        "PredictedSalary": y_pred
    })
    results.to_csv("linear_regression_predictions.csv", index=False)
    mlflow.log_artifact("linear_regression_predictions.csv")

    mlflow.sklearn.log_model(
        model,
        name="linear_regression_model"
    )

    print("===== LINEAR REGRESSION =====")
    print("R2 Score:", r2)
    print("Performance:", r2 * 100, "%")
    print("MAE:", mae)
    print("RMSE:", rmse)
    print("\nParameters logged")
    print("Metrics logged")
    print("Figure logged")
    print("Artifact logged")
    print("Model logged")
