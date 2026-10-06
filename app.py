from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "road_accident_data.csv"

app = Flask(__name__)

CATEGORICAL = [
    "Weather", "Road_Type", "Light_Condition",
    "Road_Surface", "Traffic_Density", "Alcohol_Involvement"
]
NUMERIC = ["Speed_kmph", "Vehicles_Involved", "Visibility_km"]


def train_model():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE.name}. "
            "Copy your existing road_accident_data.csv into this folder."
        )

    df = pd.read_csv(DATA_FILE)

    required = CATEGORICAL + NUMERIC + ["Severity"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            "Dataset is missing required columns: " + ", ".join(missing)
        )

    X = df.drop(columns=["Severity"])
    y = df["Severity"]

    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("num", "passthrough", NUMERIC)
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=150,
            random_state=42,
            max_depth=10
        ))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    return model, df, accuracy


MODEL = None
DATA = None
ACCURACY = None
STARTUP_ERROR = None

try:
    MODEL, DATA, ACCURACY = train_model()
except Exception as exc:
    STARTUP_ERROR = str(exc)


@app.route("/")
def index():
    return render_template("index.html")


@app.get("/api/status")
def status():
    if STARTUP_ERROR:
        return jsonify({"ready": False, "error": STARTUP_ERROR}), 500

    return jsonify({
        "ready": True,
        "rows": int(len(DATA)),
        "accuracy": round(ACCURACY * 100, 2),
        "classes": [str(x) for x in MODEL.named_steps["classifier"].classes_]
    })


@app.post("/api/predict")
def predict():
    if STARTUP_ERROR:
        return jsonify({"error": STARTUP_ERROR}), 500

    data = request.get_json(silent=True) or {}

    try:
        record = {
            "Weather": str(data["Weather"]),
            "Road_Type": str(data["Road_Type"]),
            "Light_Condition": str(data["Light_Condition"]),
            "Road_Surface": str(data["Road_Surface"]),
            "Speed_kmph": float(data["Speed_kmph"]),
            "Vehicles_Involved": int(data["Vehicles_Involved"]),
            "Visibility_km": float(data["Visibility_km"]),
            "Traffic_Density": str(data["Traffic_Density"]),
            "Alcohol_Involvement": str(data["Alcohol_Involvement"])
        }

        if record["Speed_kmph"] <= 0:
            raise ValueError("Speed must be greater than 0.")
        if record["Vehicles_Involved"] <= 0:
            raise ValueError("Vehicles involved must be greater than 0.")
        if record["Visibility_km"] <= 0:
            raise ValueError("Visibility must be greater than 0.")

        input_df = pd.DataFrame([record])
        prediction = str(MODEL.predict(input_df)[0])
        probabilities = MODEL.predict_proba(input_df)[0]
        classes = MODEL.named_steps["classifier"].classes_

        confidence = {
            str(label): round(float(prob) * 100, 2)
            for label, prob in zip(classes, probabilities)
        }

        return jsonify({
            "prediction": prediction,
            "probabilities": confidence,
            "model_accuracy": round(ACCURACY * 100, 2)
        })

    except KeyError as exc:
        return jsonify({"error": f"Missing input field: {exc.args[0]}"}), 400
    except (ValueError, TypeError) as exc:
        return jsonify({"error": str(exc)}), 400


@app.get("/api/stats")
def stats():
    if STARTUP_ERROR:
        return jsonify({"error": STARTUP_ERROR}), 500

    severity = (
        DATA["Severity"]
        .astype(str)
        .value_counts()
        .to_dict()
    )

    weather = (
        DATA["Weather"]
        .astype(str)
        .value_counts()
        .head(6)
        .to_dict()
    )

    return jsonify({
        "total_accidents": int(len(DATA)),
        "severity": {str(k): int(v) for k, v in severity.items()},
        "weather": {str(k): int(v) for k, v in weather.items()},
        "avg_speed": round(float(DATA["Speed_kmph"].mean()), 1),
        "avg_visibility": round(float(DATA["Visibility_km"].mean()), 1),
        "avg_vehicles": round(float(DATA["Vehicles_Involved"].mean()), 1)
    })


if __name__ == "__main__":
    print("=" * 62)
    print("ROAD ACCIDENT SEVERITY PREDICTOR - WEB DASHBOARD")
    print("=" * 62)
    if STARTUP_ERROR:
        print("Startup error:", STARTUP_ERROR)
    else:
        print(f"Dataset rows : {len(DATA)}")
        print(f"Test accuracy: {ACCURACY * 100:.2f}%")
        print("Open         : http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
