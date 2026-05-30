# coding:utf-8
# author: ShuoFeng

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import os

top_num = 4  # Number of top predictions to retrieve

# 1. Load the data
data = pd.read_csv('data.csv', encoding='GB2312')

# 2. Define features and target variable
features = ['Elevation', 'Temperature', 'Rainfall', 'Latitude and Longitude',
            'Slope gradient', 'Canopy density', 'Soil thickness', 'Humus thickness',
            'Slope direction', 'Slope position']

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

# 6. Load the saved model
model_save_path = "tree_species_better.h5"

if os.path.exists(model_save_path):
    loaded_model = load_model(model_save_path)
    print("Model has been successfully loaded.")

    # **Predict on test data**
    y_pred_probs_test = loaded_model.predict(X_test_scaled)
    top_pred_test = pd.DataFrame(
        [y_pred_probs_test[i].argsort()[-1 * top_num:][::-1] for i in range(y_pred_probs_test.shape[0])],
        columns=[f"Top_{i + 1}" for i in range(top_num)]
    )
    top_pred_test = top_pred_test.applymap(lambda x: y_encoded.columns[x])
    top_pred_test['Species Code'] = y_test.idxmax(axis=1).reset_index(drop=True)
    test_results = pd.DataFrame({
        'Species Code': top_pred_test['Species Code'],
        'Dataset': 'Test Set',
        **{f"Top_{i + 1}": top_pred_test[f"Top_{i + 1}"] for i in range(top_num)}  # Dynamically add Top_N columns
    })

    # Add original feature values
    for feature in features:
        test_results[f'Original_{feature}'] = data.iloc[X_test.index][f'Original_{feature}'].values

    # **Predict on training data**
    y_pred_probs_train = loaded_model.predict(X_train_scaled)
    top_pred_train = pd.DataFrame(
        [y_pred_probs_train[i].argsort()[-1 * top_num:][::-1] for i in range(y_pred_probs_train.shape[0])],
        columns=[f"Top_{i + 1}" for i in range(top_num)]
    )
    top_pred_train = top_pred_train.applymap(lambda x: y_encoded.columns[x])
    top_pred_train['Species Code'] = y_train.idxmax(axis=1).reset_index(drop=True)
    train_results = pd.DataFrame({
        'Species Code': top_pred_train['Species Code'],
        'Dataset': 'Train Set',
        **{f"Top_{i + 1}": top_pred_train[f"Top_{i + 1}"] for i in range(top_num)}  # Dynamically add Top_N columns
    })

    # Add original feature values
    for feature in features:
        train_results[f'Original_{feature}'] = data.iloc[X_train.index][f'Original_{feature}'].values

    # **Concatenate the training and test results**
    combined_results = pd.concat([train_results, test_results], ignore_index=True)

    # Save the results to a CSV file
    output_file = './top'+str(top_num)+'.csv'
    combined_results.to_csv(output_file, index=False, encoding='utf-8-sig')
