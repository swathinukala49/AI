import pandas as pd
import mlflow
import mlflow.keras
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from tensorflow import keras
from tensorflow.keras import layers


mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Salary Prediction")

data = pd.read_csv("Salary_Data.csv")
X = data[["YearsExperience"]]
y = data["Salary"]

test_size = 0.2
random_state = 42
epochs = 100
batch_size = 8
learning_rate = 0.001

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=test_size,
    random_state=random_state
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = keras.Sequential([
    layers.Input(shape=(1,)),
    layers.Dense(64, activation="relu"),
    layers.Dense(32, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(1)
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
    loss="mse"
)

with mlflow.start_run(run_name="Deep Learning"):

    history = model.fit(
        X_train_scaled,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0
    )

    y_pred = model.predict(X_test_scaled, verbose=0).flatten()

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5

    mlflow.log_params({
        "test_size": test_size,
        "random_state": random_state,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "hidden_layers": 3
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
    plt.title("Deep Learning - Actual vs Predicted")
    plt.tight_layout()
    mlflow.log_figure(plt.gcf(), "deep_learning_prediction.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Deep Learning Training Loss")
    plt.tight_layout()
    mlflow.log_figure(plt.gcf(), "training_loss.png")
    plt.close()

    results = pd.DataFrame({
        "YearsExperience": X_test["YearsExperience"].values,
        "ActualSalary": y_test.values,
        "PredictedSalary": y_pred
    })
    results.to_csv("deep_learning_predictions.csv", index=False)
    mlflow.log_artifact("deep_learning_predictions.csv")

    joblib.dump(scaler, "scaler.pkl")
    mlflow.log_artifact("scaler.pkl")
    mlflow.keras.log_model(model, name="deep_learning_model")

    print("\n===== DEEP LEARNING =====")
    print("R2 Score:", r2)
    print("Performance:", r2 * 100, "%")
    print("MAE:", mae)
    print("RMSE:", rmse)
    print("Parameters logged")
    print("Metrics logged")
    print("Figures logged")
    print("Artifacts logged")
    print("Model logged")
