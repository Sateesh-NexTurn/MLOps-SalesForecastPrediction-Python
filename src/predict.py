import pandas as pd
import pickle

# Load model
model_path = 'models/sales_model.pkl'

with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Future data for prediction
future_data = pd.DataFrame({
    'Store': [1],
    'Product': [0],
    'Year': [2026],
    'Month': [6],
    'Day': [15],
    'Weekday': [1]
})

# Predict sales
prediction = model.predict(future_data)

print("Predicted Sales:")
print(prediction[0])