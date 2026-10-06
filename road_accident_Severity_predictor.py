"""
ROAD ACCIDENT SEVERITY PREDICTOR
Academic Machine Learning Mini Project

This script:
1. Loads road_accident_data.csv
2. Preprocesses categorical variables using One-Hot Encoding
3. Trains a Random Forest classifier
4. Prints evaluation metrics
5. Saves confusion_matrix.png
6. Demonstrates a sample prediction with probabilities

Expected CSV columns:
Weather, Road_Type, Light_Condition, Road_Surface, Speed_kmph,
Vehicles_Involved, Visibility_km, Traffic_Density, Alcohol_Involvement, Severity
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

BASE = Path(__file__).resolve().parent
DATA_FILE = BASE / "road_accident_data.csv"

CATEGORICAL = [
    "Weather", "Road_Type", "Light_Condition",
    "Road_Surface", "Traffic_Density", "Alcohol_Involvement"
]
NUMERIC = ["Speed_kmph", "Vehicles_Involved", "Visibility_km"]
TARGET = "Severity"

def build_model():
    df = pd.read_csv(DATA_FILE)
    required = CATEGORICAL + NUMERIC + [TARGET]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("Missing columns: " + ", ".join(missing))

    X = df[CATEGORICAL + NUMERIC]
    y = df[TARGET].astype(str)

    preprocessor = ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("numeric", "passthrough", NUMERIC)
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=150, random_state=42, max_depth=10
        ))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    model.fit(X_train, y_train)
    return df, model, X_test, y_test

def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            "road_accident_data.csv was not found next to this Python file."
        )

    df, model, X_test, y_test = build_model()
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)

    print("=" * 65)
    print("       ROAD ACCIDENT SEVERITY PREDICTOR")
    print("=" * 65)
    print(f"Dataset records       : {len(df)}")
    print(f"Training records      : {len(df) - len(X_test)}")
    print(f"Testing records       : {len(X_test)}")
    print("Algorithm             : Random Forest Classifier")
    print("Number of trees       : 150")
    print("Maximum tree depth    : 10")
    print("Preprocessing         : One-Hot Encoding")
    print(f"Test accuracy         : {acc * 100:.2f}%")
    print("\nCLASSIFICATION REPORT")
    print("-" * 65)
    print(classification_report(y_test, pred, zero_division=0))

    cm = confusion_matrix(y_test, pred, labels=model.named_steps["classifier"].classes_)
    fig, ax = plt.subplots(figsize=(7, 5))
    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=model.named_steps["classifier"].classes_
    ).plot(ax=ax)
    ax.set_title("Road Accident Severity - Confusion Matrix")
    fig.tight_layout()
    fig.savefig(BASE / "confusion_matrix.png", dpi=180)
    plt.close(fig)

    sample = pd.DataFrame([{
        "Weather": "Rain",
        "Road_Type": "Highway",
        "Light_Condition": "Night",
        "Road_Surface": "Wet",
        "Speed_kmph": 85,
        "Vehicles_Involved": 3,
        "Visibility_km": 3,
        "Traffic_Density": "High",
        "Alcohol_Involvement": "Yes"
    }])

    result = model.predict(sample)[0]
    probs = model.predict_proba(sample)[0]
    classes = model.named_steps["classifier"].classes_

    print("\nSAMPLE PREDICTION")
    print("-" * 65)
    print("Weather              : Rain")
    print("Road Type            : Highway")
    print("Light Condition      : Night")
    print("Road Surface         : Wet")
    print("Speed                : 85 km/h")
    print("Vehicles Involved    : 3")
    print("Visibility           : 3 km")
    print("Traffic Density      : High")
    print("Alcohol Involvement  : Yes")
    print(f"\nPredicted Severity   : {result}")
    print("\nClass probabilities:")
    for label, probability in zip(classes, probs):
        print(f"  {label:<12}: {probability * 100:.2f}%")

if __name__ == "__main__":
    main()
