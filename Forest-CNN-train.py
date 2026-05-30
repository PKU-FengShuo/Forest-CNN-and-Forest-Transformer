# coding:utf-8
# author: ShuoFeng

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten
import os
import numpy as np

# 1. Load the dataset
data = pd.read_csv('data.csv', encoding='GB2312')

# 2. Define features and target variable
features = ['Elevation', 'Temperature', 'Rainfall', 'Latitude and Longitude',
            'Slope gradient', 'Canopy density', 'Soil thickness', 'Humus thickness',
            'Slope direction', 'Slope position']
target = 'Tree_Species'

# Separate features and target variable
X = data[features]
y = data[target]

# 3. One-Hot encode the target variable
y_encoded = pd.get_dummies(y)

# 4. Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, shuffle=True)

# 5. Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Build the model
input_dim = X_train_scaled.shape[1]  # Number of features
output_dim = y_encoded.shape[1]  # Number of unique tree species codes

model = Sequential([
    Conv1D(64, 3, activation='relu', input_shape=(input_dim, 1)),  # First Convolutional layer
    Flatten(),  # Flatten the output for fully connected layers
    Dense(256, activation='relu'),  # First fully connected layer
    Dense(128, activation='relu'),  # Second fully connected layer
    Dense(64, activation='relu'),   # Third fully connected layer
    Dense(output_dim, activation='softmax')  # Output layer with softmax for classification
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Reshape the input data to fit the Conv1D layer requirements
X_train_scaled = X_train_scaled[..., np.newaxis]  # Add an extra dimension for Conv1D
X_test_scaled = X_test_scaled[..., np.newaxis]

# 7. Train the model
model.fit(X_train_scaled, y_train, epochs=10000, batch_size=32, validation_split=0.2, verbose=1, shuffle=True)

# 8. Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"Test accuracy: {accuracy:.4f}")

# 9. Save the trained model
model_save_path = "tree_species_better.h5"
model.save(model_save_path)
print(f"Model has been saved to: {model_save_path}")
