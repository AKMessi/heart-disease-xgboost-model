from heart_disease_final import CalibratedModel
import joblib
import pandas as pd

model = joblib.load('heart_model_calibrated.pkl')
features  = joblib.load('feature_cols.pkl')
threshold = joblib.load('threshold.pkl')

patient = pd.DataFrame([{
      'age': 58, 'sex': 1, 'bmi': 29.5,
      'smoking_current': 1, 'diabetes': 1,
      'gen_health': 3, 'stroke': 0,
      # fill in what you have, leave the rest as NaN
}])
  # add any missing columns as NaN
for col in features:
      if col not in patient.columns:
          patient[col] = float('nan')

risk  = model.predict_proba(patient[features])[0, 1]
alert = risk >= threshold
print(f'Risk score : {risk:.1%}')
print(f'High risk  : {alert}')