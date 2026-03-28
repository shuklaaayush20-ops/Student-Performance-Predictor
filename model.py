# model.py
# This file handles everything related to our ML model -
# loading data, training the decision tree, saving it, and making predictions.
# I also added some charts at the end to visualize the dataset.

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # needed so plots save to file instead of opening a window


# -------------------------------------------------
# Step 1: Load the dataset
# -------------------------------------------------
def load_data(filepath="data.csv"):
    # just reads the csv file and returns a dataframe
    df = pd.read_csv(filepath)
    print(f"Dataset loaded successfully: {df.shape[0]} rows and {df.shape[1]} columns")
    return df


# -------------------------------------------------
# Step 2: Prepare data for training
# -------------------------------------------------
def preprocess(df):
    # The 'performance' column has text values like Good, Average, Poor
    # but ML models need numbers, so we use LabelEncoder to convert them
    le = LabelEncoder()
    df["performance_encoded"] = le.fit_transform(df["performance"])

    # by default sklearn sorts alphabetically: Average=0, Good=1, Poor=2
    print(f"Classes found: {list(le.classes_)}")

    # these are our input features
    feature_cols = ["study_hours", "sleep_hours", "attendance", "previous_marks"]
    X = df[feature_cols]
    y = df["performance_encoded"]

    return X, y, le


# -------------------------------------------------
# Step 3: Train the Decision Tree model
# -------------------------------------------------
def train_model(X, y):
    # splitting into 80% training and 20% testing
    # stratify=y makes sure all 3 classes appear in both splits
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # using max_depth=5 so the tree doesn't overfit on training data
    model = DecisionTreeClassifier(max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    # checking how well it does on the test set
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nModel Accuracy on test data: {accuracy * 100:.2f}%\n")
    print("Detailed Classification Report:")
    print(classification_report(y_test, y_pred))

    return model, X_test, y_test, y_pred, accuracy


# -------------------------------------------------
# Step 4: Save the trained model to disk
# -------------------------------------------------
def save_artifacts(model, le, model_path="model.pkl", encoder_path="encoder.pkl"):
    # saving using pickle so we don't have to retrain every time
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(encoder_path, "wb") as f:
        pickle.dump(le, f)

    print(f"Model saved: {model_path}")
    print(f"Encoder saved: {encoder_path}")


# -------------------------------------------------
# Step 5: Load saved model back from disk
# -------------------------------------------------
def load_artifacts(model_path="model.pkl", encoder_path="encoder.pkl"):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(encoder_path, "rb") as f:
        le = pickle.load(f)
    return model, le


# -------------------------------------------------
# Step 6: Predict performance for a new student
# -------------------------------------------------
def predict_performance(study_hours, sleep_hours, attendance, previous_marks,
                        model=None, le=None):
    # load from file if model isn't passed directly
    if model is None or le is None:
        model, le = load_artifacts()

    # putting the inputs into a dataframe so sklearn can process it
    input_data = pd.DataFrame(
        [[study_hours, sleep_hours, attendance, previous_marks]],
        columns=["study_hours", "sleep_hours", "attendance", "previous_marks"]
    )

    encoded_pred = model.predict(input_data)[0]
    label = le.inverse_transform([encoded_pred])[0]  # convert number back to text
    return label


# -------------------------------------------------
# Step 7: Generate some basic charts for the report
# -------------------------------------------------
def generate_visualizations(df, output_dir="."):
    os.makedirs(output_dir, exist_ok=True)

    # color coding: green=Good, orange=Average, red=Poor
    colors = {"Good": "#2ecc71", "Average": "#f39c12", "Poor": "#e74c3c"}

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("Student Performance - Dataset Analysis", fontsize=16, fontweight="bold", y=1.01)

    # Chart 1 - how many students fall in each category
    counts = df["performance"].value_counts()
    axes[0, 0].pie(
        counts,
        labels=counts.index,
        autopct="%1.1f%%",
        colors=[colors.get(k, "#999") for k in counts.index],
        startangle=140,
        wedgeprops={"edgecolor": "white", "linewidth": 2}
    )
    axes[0, 0].set_title("Performance Distribution")

    # Chart 2 - does studying more lead to better marks? (scatter)
    for perf, group in df.groupby("performance"):
        axes[0, 1].scatter(
            group["study_hours"], group["previous_marks"],
            label=perf, color=colors[perf], alpha=0.7, edgecolors="white", s=60
        )
    axes[0, 1].set_xlabel("Study Hours")
    axes[0, 1].set_ylabel("Previous Marks")
    axes[0, 1].set_title("Study Hours vs Previous Marks")
    axes[0, 1].legend()
    axes[0, 1].grid(True, linestyle="--", alpha=0.4)

    # Chart 3 - average value of each feature for each performance group
    feature_means = df.groupby("performance")[
        ["study_hours", "sleep_hours", "attendance", "previous_marks"]
    ].mean()
    feature_means.plot(kind="bar", ax=axes[1, 0], colormap="Set2", edgecolor="white")
    axes[1, 0].set_title("Average Feature Values by Performance Category")
    axes[1, 0].set_xlabel("Performance Category")
    axes[1, 0].set_ylabel("Average Value")
    axes[1, 0].legend(fontsize=8, loc="upper right")
    axes[1, 0].tick_params(axis="x", rotation=0)
    axes[1, 0].grid(axis="y", linestyle="--", alpha=0.4)

    # Chart 4 - attendance spread across performance groups (box plot)
    perf_order = ["Good", "Average", "Poor"]
    data_boxes = [df[df["performance"] == p]["attendance"].values for p in perf_order]
    bp = axes[1, 1].boxplot(
        data_boxes,
        tick_labels=perf_order,
        patch_artist=True,
        medianprops={"color": "white", "linewidth": 2}
    )
    for patch, perf in zip(bp["boxes"], perf_order):
        patch.set_facecolor(colors[perf])
        patch.set_alpha(0.8)
    axes[1, 1].set_title("Attendance Distribution by Performance")
    axes[1, 1].set_xlabel("Performance Category")
    axes[1, 1].set_ylabel("Attendance (%)")
    axes[1, 1].grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    chart_path = os.path.join(output_dir, "visualizations.png")
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Charts saved to: {chart_path}")
    return chart_path


# -------------------------------------------------
# Step 8: Run everything in order
# -------------------------------------------------
def main():
    df = load_data("data.csv")
    X, y, le = preprocess(df)
    model, X_test, y_test, y_pred, accuracy = train_model(X, y)
    save_artifacts(model, le)
    generate_visualizations(df)

    # saving accuracy so app.py can display it without retraining
    with open("accuracy.txt", "w") as f:
        f.write(str(round(accuracy * 100, 2)))

    print("\nAll done! Now run: streamlit run app.py")
    return accuracy


if __name__ == "__main__":
    main()
