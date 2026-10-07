import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import mlflow
import mlflow.keras

import matplotlib.pyplot as plt
import pandas as pd


# ==========================================
# MLFLOW SETUP
# ==========================================

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("IMDB Sentiment Analysis")


# ==========================================
# PARAMETERS
# ==========================================

num_words = 10000
maxlen = 200
embedding_dim = 64
lstm_units = 64
dense_units = 64

epochs = 10
batch_size = 64
validation_split = 0.2


# ==========================================
# LOAD IMDB DATASET
# ==========================================

(train_data, train_labels), (test_data, test_labels) = \
    keras.datasets.imdb.load_data(num_words=num_words)


# ==========================================
# PADDING
# ==========================================

train_data = keras.preprocessing.sequence.pad_sequences(
    train_data,
    maxlen=maxlen,
    truncating="post",
    padding="post"
)

test_data = keras.preprocessing.sequence.pad_sequences(
    test_data,
    maxlen=maxlen,
    truncating="post",
    padding="post"
)


# ==========================================
# BUILD LSTM MODEL
# ==========================================

model = keras.Sequential([
    layers.Embedding(
        input_dim=num_words,
        output_dim=embedding_dim
    ),

    layers.LSTM(lstm_units),

    layers.Dense(
        dense_units,
        activation="relu"
    ),

    layers.Dense(
        1,
        activation="sigmoid"
    )
])


model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)


model.summary()


# ==========================================
# MLFLOW RUN
# ==========================================

with mlflow.start_run(run_name="LSTM Sentiment Analysis"):

    # ======================================
    # TRAIN MODEL
    # ======================================

    history = model.fit(
        train_data,
        train_labels,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        verbose=1
    )


    # ======================================
    # TEST MODEL
    # ======================================

    test_loss, test_accuracy = model.evaluate(
        test_data,
        test_labels,
        verbose=0
    )


    # ======================================
    # LOG PARAMETERS
    # ======================================

    mlflow.log_params({
        "num_words": num_words,
        "maxlen": maxlen,
        "embedding_dim": embedding_dim,
        "lstm_units": lstm_units,
        "dense_units": dense_units,
        "epochs": epochs,
        "batch_size": batch_size,
        "validation_split": validation_split
    })


    # ======================================
    # LOG METRICS
    # ======================================

    mlflow.log_metrics({
        "test_accuracy": test_accuracy,
        "test_loss": test_loss,
        "final_train_accuracy":
            history.history["accuracy"][-1],
        "final_val_accuracy":
            history.history["val_accuracy"][-1],
        "final_train_loss":
            history.history["loss"][-1],
        "final_val_loss":
            history.history["val_loss"][-1]
    })


    # ======================================
    # FIGURE 1 - ACCURACY
    # ======================================

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
    plt.title("LSTM Training and Validation Accuracy")
    plt.legend()

    plt.tight_layout()

    mlflow.log_figure(
        plt.gcf(),
        "accuracy_plot.png"
    )

    plt.close()


    # ======================================
    # FIGURE 2 - LOSS
    # ======================================

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
    plt.ylabel("Loss")
    plt.title("LSTM Training and Validation Loss")
    plt.legend()

    plt.tight_layout()

    mlflow.log_figure(
        plt.gcf(),
        "loss_plot.png"
    )

    plt.close()


    # ======================================
    # LOG TRAINING HISTORY AS ARTIFACT
    # ======================================

    history_df = pd.DataFrame(history.history)

    history_df.to_csv(
        "training_history.csv",
        index=False
    )

    mlflow.log_artifact(
        "training_history.csv"
    )


    # ======================================
    # LOG MODEL
    # ======================================

    mlflow.keras.log_model(
        model,
        name="lstm_sentiment_model"
    )


    # ======================================
    # PRINT RESULTS
    # ======================================

    print("\n==============================")
    print("LSTM SENTIMENT ANALYSIS")
    print("==============================")

    print("Test Accuracy:", test_accuracy)
    print("Test Loss:", test_loss)

    print("\nParameters logged ✅")
    print("Metrics logged ✅")
    print("Figures logged ✅")
    print("Artifacts logged ✅")
    print("Model logged ✅")
