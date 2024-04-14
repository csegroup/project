import pickle
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import time
import concurrent.futures
import os

start_time = time.time()

def read_csv():
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    csv_path = os.path.join(parent_dir, 'moscow_with_distances.csv')
    data = pd.read_csv(csv_path, sep=';')
    """
    data = pd.read_csv("moscow_with_distances.csv")
    return data

with concurrent.futures.ThreadPoolExecutor() as executor:
    data_future = executor.submit(read_csv)
    data = data_future.result()

# Drop irrelevant columns if any
data = data[['price', 'rooms', 'area', 'level', 'levels', 'kitchen_area', 'building_type', 'object_type', 'Distance']]

# Handle missing values if any
data = data.dropna()

# Split data into features and target variable
X = data[['rooms', 'area', 'level', 'levels', 'kitchen_area', 'building_type', 'object_type','Distance']].values
y = data['price'].values


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Note: Random Forest doesn't require feature scaling, but it's good practice to scale your features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the model
random_forest_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
random_forest_model.fit(X_train_scaled, y_train)

# Predictions
predictions = random_forest_model.predict(X_test_scaled)

# Calculate and print evaluation metrics
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

# Save the RandomForest model and the scaler
model_data = {
    'model': random_forest_model,
    'scaler': scaler
}

# Serialize and save to a .pth file
with open('random_forest_model.pth', 'wb') as f:
    pickle.dump(model_data, f)

print("Model and scaler have been saved.")

"""
# Predicting a house with specific features
new_house_features = np.array([[2, 35, 3, 10, 8, 4, 2]])  
new_house_features_scaled = scaler.transform(new_house_features)
predicted_price = random_forest_model.predict(new_house_features_scaled)
print(f"Predicted Price: {predicted_price[0]:.2f}")
"""

# Record the end time
end_time = time.time()

# Calculate the total execution time
execution_time = end_time - start_time
print("Total Execution Time: {:.2f} seconds".format(execution_time))