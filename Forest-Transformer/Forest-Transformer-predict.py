# coding:utf-8
# author: ShuoFeng

# Import necessary modules
import tensorflow as tf
import numpy as np
import pandas as pd
import os


# Define TransformerBlock class
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [tf.keras.layers.Dense(ff_dim, activation='relu'), tf.keras.layers.Dense(embed_dim)]
        )
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


# Define Forest-Transformer model class
class ForestTransformer(tf.keras.Model):
    def __init__(self):
        super(ForestTransformer, self).__init__()
        self.f1 = tf.keras.layers.Dense(64, activation='relu', kernel_regularizer='l2')
        self.f2 = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer='l2')
        self.transformer = TransformerBlock(embed_dim=3, num_heads=2, ff_dim=32)
        self.flatten = tf.keras.layers.Flatten()
        self.f3 = tf.keras.layers.Dense(256, activation='relu', kernel_regularizer='l2')
        self.f4 = tf.keras.layers.Dense(64, activation='relu', kernel_regularizer='l2')
        self.f5 = tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer='l2')

    def call(self, x):
        static_data = x[:, :-3]  # Static data
        time_series_data = x[:, -3:]  # Time series data
        time_series_data = tf.expand_dims(time_series_data, axis=1)  # Reshape for transformer

        transformer_output = self.transformer(time_series_data)
        combined_features = tf.concat([static_data, self.flatten(transformer_output)], axis=-1)

        x = self.f3(combined_features)
        x = self.f4(x)
        y = self.f5(x)
        y = 20 * y  # Scale output
        return y


# Load data
df = pd.read_csv('./2023_predict.csv', encoding='GB2312')

# Data preprocessing
df['DBH9'] = df['DBH9'] / 100
df['DBH8'] = df['DBH8'] / 100
df['DBH7'] = df['DBH7'] / 100

keys = ['Tree species', 'Elevation', 'Temperature', 'Rainfall', 'Latitude ',' Longitude',
        'Slope gradient', 'Canopy density', 'Soil thickness', 'Humus thickness',
        'Slope direction', 'Slope position', 'DBH7', 'DBH8', 'DBH9']

x = np.array(df[keys])

# Convert to tensor
x = tf.convert_to_tensor(x, dtype=tf.float32)

# Load model
ckpt_addr = 'transformer_time'
checkpoint_save_path_num = f"./ckpt/{ckpt_addr}_rate.ckpt"
model = ForestTransformer()

if os.path.exists(checkpoint_save_path_num + '.index'):
    print('-------------Loading the model-----------------')
    model.load_weights(checkpoint_save_path_num)
else:
    print('No checkpoint found! Please ensure the model has been trained and saved properly.')

# Loop to predict future DBH values
num_years_to_predict = (2053 - 2023) // 5  # Number of predictions to make
predicted_diameters = []  # To store predicted diameters

# Initialize the latest DBH values (normalized)
latest_diameters = df[['DBH7','DBH8', 'DBH9']].values.tolist()
shape_0 = len(latest_diameters)

latest_diameters = np.array(latest_diameters)
latest_diameters = latest_diameters.reshape(shape_0, 3)

x_ = x[:, :-3]
for _ in range(num_years_to_predict):
    # Concatenate latest DBH data with input features
    input_data = np.concatenate([x_, latest_diameters], axis=1)  # Update input features
    print(input_data[0])
    input_tensor = tf.convert_to_tensor(input_data, dtype=tf.float32)

    # Predict new DBH growth (normalized values)
    prediction = model.predict(input_tensor).flatten()  # Extract prediction values

    # Compute new DBH values (scaled back to original units)
    new_diameter = latest_diameters[:, -1] * 100 + prediction  # Compute new DBH (DBH11, DBH12, ...)

    # Update DBH list (and ensure normalization)
    predicted_diameters.append(new_diameter)
    latest_diameters = np.array(
        [latest_diameters[:, 1], latest_diameters[:, 2], new_diameter / 100])  # Sliding window update
    latest_diameters = latest_diameters.T

# Add predicted diameters to the dataframe
for i, diameter in enumerate(predicted_diameters, start=11):
    df[f'DBH{i}'] = diameter

# Save the final results to a CSV file
df.to_csv('result.csv', index=False, encoding='utf-8-sig')