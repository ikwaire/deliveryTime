# Delivery Time Estimation

**NextGen 3MTT — Data Science Track — Capstone Project**

## Problem

Logistics ETAs are unreliable. Customers don't know when their order will
actually arrive, and dispatch/ops teams can't plan routes, riders, or
customer-facing promises around a trustworthy number.

## What this project builds (MVP)

A machine learning model that predicts **delivery time in minutes** for a
delivery order, based on route, order, environmental, and rider features —
plus an honest evaluation of *where* the model is accurate and where it isn't,
so the estimate can be shown with an appropriate confidence buffer.

## Repository structure

```
delivery_time_estimation/
├── README.md                          <- you are here
├── requirements.txt                   <- Python dependencies
├── notebook/
│   ├── delivery_time_estimation.ipynb <- main notebook (run this)
│   └── delivery_time_estimation.py    <- same notebook as a plain .py script
│                                          (jupytext "percent" format — easier
│                                          to review/diff in git)
├── data/
│   ├── delivery_orders.csv            <- generated dataset (6,000 orders)
│   └── *.png                          <- EDA / evaluation charts saved by the notebook
└── models/
    ├── delivery_time_model.pkl        <- final trained model (preprocessing + model, one artifact)
    └── model_metadata.json            <- which model won, and its test metrics
```

## Dataset

No public real-world delivery dataset with the right fields was available for
this capstone, so the notebook **simulates a realistic dataset** (6,000
orders) modeled on last-mile delivery conditions in Kebbi State / North-West
Nigeria: a Birnin Kebbi hub dispatching to towns like Bunza, Argungu, Kamba,
Zuru, Jega, Yauri and Aliero, with realistic route, rider and environmental
features:

| Feature | Description |
|---|---|
| `distance_km` | Route distance |
| `vehicle_type` | Motorcycle, Keke (Tricycle), Van, Truck |
| `road_condition` | Paved / Unpaved |
| `weather` | Clear / Rain / Harmattan Dust |
| `traffic_level` | Low / Medium / High |
| `order_type` | Parcel, Food, Documents, Bulk Goods |
| `num_stops` | Extra stops before final drop-off (multi-drop) |
| `package_weight_kg` | Package weight |
| `rider_experience_years` | Rider's years of experience |
| `time_of_day`, `day_of_week`, `is_weekend` | When the order was placed |
| `network_signal` | Strong / Weak / No Signal — models patchy rural connectivity and its effect on dispatch confirmation delay |
| `delivery_time_minutes` | **Target** — total delivery time |

The target is generated from a realistic underlying formula (speed by
vehicle × road × weather × traffic, handling time per stop, rider-experience
effect, etc.) plus random noise — so the relationships are genuine but the
data is not perfectly clean or linear, similar to real operations data.

**To use real data instead:** replace `data/delivery_orders.csv` with a real
export that has the same column names (or update the `numeric_features` /
`categorical_features` lists in the notebook), and re-run — the rest of the
pipeline is unchanged.

## Approach

1. **Data prep** — quality checks (missing values, duplicates, types), then a
   `ColumnTransformer` (`StandardScaler` for numeric features, `OneHotEncoder`
   for categorical features) so training-time and inference-time
   preprocessing are guaranteed identical.
2. **Modeling** — four models compared with 5-fold cross-validation on the
   training set, then a final check on a held-out test set:
   - Linear Regression (baseline)
   - Ridge Regression
   - Random Forest Regressor
   - Gradient Boosting Regressor
3. **Error analysis** — residual plots, and Mean Absolute Error broken down
   by distance bucket, weather, road condition, and vehicle type, to show
   *where* the model is trustworthy and where it isn't.
4. **Evaluation** — MAE, RMSE, R² on the untouched test set; feature
   importance for the winning model.
5. **Export** — best pipeline (preprocessing + model) saved as one
   `.pkl` file with `joblib`, plus a `model_metadata.json` summary.

## Results (this run)

| Model | 5-fold CV MAE (min) |
|---|---|
| Linear Regression | ~50.1 |
| Ridge Regression | ~50.1 |
| Random Forest | ~14.4 |
| **Gradient Boosting (best)** | **~12.2** |

**Best model on held-out test set — Gradient Boosting Regressor:**
- **MAE:** ≈ 11.7 minutes
- **RMSE:** ≈ 18.5 minutes
- **R²:** ≈ 0.99

Error is lowest on short, paved, clear-weather deliveries, and highest on
long, unpaved, bad-weather routes — consistent with real-world intuition, and
a concrete basis for widening the shown ETA buffer specifically on those
harder trips instead of quoting a false level of precision everywhere.

*(Exact numbers will vary slightly between runs since the dataset is
simulated with a fixed random seed for reproducibility, but you'll see the
same ranking of models and the same error patterns.)*

## How to run

```bash
# 1. Create environment
pip install -r requirements.txt

# 2. Run the notebook
jupyter notebook notebook/delivery_time_estimation.ipynb
# (Run All Cells — it regenerates the dataset, trains all models,
#  produces the charts in data/, and saves the model in models/)
```

Or open `notebook/delivery_time_estimation.ipynb` directly in Google Colab
(upload the whole folder, or mount Google Drive, so the relative `../data`
and `../models` paths resolve).

### Using the saved model on a new order

```python
import joblib
import pandas as pd

model = joblib.load("models/delivery_time_model.pkl")

order = pd.DataFrame([{
    "destination_town": "Bunza", "distance_km": 45.0, "vehicle_type": "Motorcycle",
    "road_condition": "Unpaved", "weather": "Rain", "traffic_level": "Medium",
    "order_type": "Parcel", "num_stops": 1, "package_weight_kg": 3.5,
    "rider_experience_years": 2.0, "time_of_day": "Afternoon", "day_of_week": "Wednesday",
    "is_weekend": False, "network_signal": "Weak",
}])

print(model.predict(order))  # predicted delivery time in minutes
```

## Limitations

- Data is synthetic — realistic by design, but real orders will have messier
  patterns (GPS gaps, mis-logged timestamps, rider behavior) that this
  simulation doesn't capture. **Retrain on real operations data before
  production use.**
- No live traffic/weather feed — only categorical labels are used.
- Higher error on rural/unpaved routes reflects a genuine hard case for the
  model, not a bug — this is exactly the kind of finding an error-analysis
  section is meant to surface.

## Next steps

1. Swap in real dispatch/GPS data (same schema).
2. Add live traffic/weather API features.
3. Hyperparameter-tune the winning model (`GridSearchCV` / `RandomizedSearchCV`).
4. Serve the saved model behind a small API (FastAPI/Flask) for a dispatch app.
5. Monitor live prediction error over time to catch model drift.

## Tools

Python, pandas, scikit-learn, matplotlib/seaborn, Jupyter/Colab, joblib.

## Demo video

See `demo_video_script.md` for a 2–3 minute walkthrough script covering the
problem, the notebook, the results, and the error analysis.
