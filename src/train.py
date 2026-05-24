# =====================================
# 1. Import Libraries
# =====================================
# 1. imports
import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


# =====================================
# 2. Load Dataset
# =====================================

df = pd.read_csv("../data/train.csv")

# =====================================
# 3. Data Cleaning
# =====================================

df.dropna(inplace=True)


# =====================================
# 4. Feature Engineering
# =====================================

df['Date'] = pd.to_datetime(df['Date'])

df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['Weekday'] = df['Date'].dt.weekday


# =====================================
# 5. Encode Categorical Columns
# =====================================

df['Store'] = df['Store'].astype('category').cat.codes
df['Product'] = df['Product'].astype('category').cat.codes


# =====================================
# 6. Features and Target
# =====================================

X = df[['Store', 'Product', 'Year', 'Month', 'Day', 'Weekday']]

y = df['Sales']


# =====================================
# 7. Train-Test Split
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# =====================================
# 8. Train Model
# =====================================

model = RandomForestRegressor()

model.fit(X_train, y_train)

# =====================================
# 9. SAVE MODEL  
# =====================================

import os
import pickle

os.makedirs("models", exist_ok=True)

pickle.dump(model, open("models/sales_model.pkl", "wb"))

print("Model saved successfully")