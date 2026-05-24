# =========================================
# 1. Import Libraries
# =========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


# =========================================
# 2. Load Dataset
# =========================================

df = pd.read_csv("../data/train.csv")

print(df.head())
print(df.info())
print(df.describe())


# =========================================
# 3. Data Cleaning
# =========================================

print(df.isnull().sum())

df.dropna(inplace=True)

# OR
# df.fillna(0, inplace=True)


# =========================================
# 4. Feature Engineering
# =========================================

df['Date'] = pd.to_datetime(df['Date'])

df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['Weekday'] = df['Date'].dt.weekday


# =========================================
# 5. Exploratory Data Analysis
# =========================================

plt.figure(figsize=(12,6))
plt.plot(df['Date'], df['Sales'])
plt.title("Sales Trend")
plt.show()

monthly_sales = df.groupby('Month')['Sales'].sum()

monthly_sales.plot(kind='bar')
plt.title("Monthly Sales")
plt.show()

sns.heatmap(df.corr(numeric_only=True), annot=True)
plt.show()


# =========================================
# 6. Feature Selection
# =========================================

X = df[['Year', 'Month', 'Day', 'Weekday']]

y = df['Sales']


# =========================================
# 7. Train-Test Split
# =========================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# =========================================
# 8. Build ML Model
# =========================================

model = RandomForestRegressor()

model.fit(X_train, y_train)


# =========================================
# 9. Predictions
# =========================================

predictions = model.predict(X_test)


# =========================================
# 10. Model Evaluation
# =========================================

mae = mean_absolute_error(y_test, predictions)
print("MAE:", mae)

r2 = r2_score(y_test, predictions)
print("R2 Score:", r2)


# =========================================
# 11. Visualization
# =========================================

plt.figure(figsize=(10,5))

plt.plot(y_test.values, label='Actual')
plt.plot(predictions, label='Predicted')

plt.legend()
plt.title("Actual vs Predicted Sales")
plt.show()


# =========================================
# 12. Future Forecasting
# =========================================

future_data = pd.DataFrame({
    'Year':[2026],
    'Month':[6],
    'Day':[15],
    'Weekday':[1]
})

future_sales = model.predict(future_data)

print("Predicted Sales:", future_sales)

# =========================================
13. Save Model
# =========================================

import pickle

pickle.dump(model, open('sales_model.pkl', 'wb'))

print("Model saved successfully")