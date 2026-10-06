# Road Accident Severity Predictor — Professional Web Dashboard

This dashboard uses the same model design as the original project:
- 9 input features
- One-Hot Encoding for categorical fields
- Random Forest Classifier
- 150 estimators
- random_state=42
- max_depth=10
- Outputs Minor, Moderate, or Severe plus class probabilities

## Files

- `app.py` — Flask backend and ML prediction API
- `templates/index.html` — dashboard UI
- `static/style.css` — professional styling
- `static/script.js` — dashboard interactions and API calls
- `requirements.txt` — Python dependencies

## Setup

Keep your existing `road_accident_data.csv` in the same folder as `app.py`.

Open PowerShell in this folder:

```powershell
pip install -r requirements.txt
python app.py
```

Then open:

http://127.0.0.1:5000

## Demo

1. Start `python app.py`
2. Open the URL above in Chrome
3. Dashboard Overview shows dataset/model statistics
4. Click Prediction
5. Click Load Demo Case
6. Click Predict Severity
7. Show the predicted class and probability bars
8. Open Analytics to show dataset statistics
9. Open About Model to explain the ML pipeline

## Important

This web dashboard retrains the same Random Forest pipeline from `road_accident_data.csv` when the server starts. It is therefore connected to the same dataset/model logic used by the original Tkinter predictor.

For the sample case from the project report, use:
Rain, Highway, Night, Wet, 85 km/h, 3 vehicles, 3 km visibility, High traffic, Alcohol = Yes.

The prediction can vary if the CSV or model code changes. The system is for academic demonstration only.
