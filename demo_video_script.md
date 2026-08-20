# Demo Video Script — Delivery Time Estimation (2–3 minutes)

Record your screen (OBS Studio, Loom, or Zoom "record yourself") walking
through the notebook while you talk. Suggested timing below — adjust to fit
your own pace, total should stay under 3 minutes.

---

### 1. Problem & goal (0:00 – 0:30)

> "Logistics ETAs are unreliable — customers and dispatch teams can't trust
> the delivery-time estimate they're given. For this capstone I built a
> machine learning model that predicts delivery time in minutes, and — just
> as important — I evaluated *where* the model is accurate and where it
> isn't, so the estimate can be trusted enough to actually show a customer."

Show: title markdown cell at the top of the notebook.

### 2. The data (0:30 – 1:00)

> "No real delivery dataset was available, so I simulated a realistic
> dataset of 6,000 orders modeled on last-mile delivery here in Kebbi State —
> distance, vehicle type, road condition, weather, traffic, number of stops,
> rider experience, even network signal strength, since that affects
> dispatch confirmation delay in areas with patchy connectivity."

Show: the data generation cell + `df.head()` output, and one EDA chart
(e.g. delivery time vs. distance by road condition).

### 3. Modeling (1:00 – 1:45)

> "I compared four models — Linear Regression as a baseline, Ridge
> Regression, Random Forest, and Gradient Boosting — using 5-fold
> cross-validation. Gradient Boosting won, with a mean absolute error of
> about 12 minutes on cross-validation, and about 11.7 minutes on the
> held-out test set, with an R² of 0.99."

Show: the model comparison table / cross-validation output, and the final
results_df table.

### 4. Error analysis (1:45 – 2:20)

> "A single accuracy number can hide where a model actually fails, so I
> broke the error down by distance, weather, road condition and vehicle
> type. The model is most accurate on short, paved-road, clear-weather
> deliveries, and least accurate on long, unpaved, bad-weather routes — which
> matches real-world intuition, and gives the dispatch team a concrete reason
> to widen the ETA buffer specifically on those harder trips."

Show: the predicted-vs-actual scatter plot, the residuals histogram, and the
MAE-by-slice bar charts.

### 5. Wrap-up (2:20 – 2:45)

> "The final model and preprocessing pipeline are saved as a single file, so
> it can be loaded and used to score new orders immediately — I show that
> here with a couple of example orders. Next steps are to replace the
> synthetic data with real dispatch logs, add live traffic and weather data,
> and serve the model behind an API for a dispatch app."

Show: the "predict on new orders" cell and its output table.

---

## Recording checklist

- [ ] Screen recording software ready (OBS Studio / Loom / Zoom)
- [ ] Notebook already executed once so cells run fast during recording
      (or narrate over the outputs instead of re-running live)
- [ ] Close unrelated tabs/notifications before recording
- [ ] Keep to 2–3 minutes — practice once with a timer
- [ ] Export as .mp4 and upload alongside the notebook/repo
