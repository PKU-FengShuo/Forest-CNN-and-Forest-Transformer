# coding:utf-8
# author: ShuoFeng

# 1. Import required modules
import tensorflow as tf
import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization, MultiHeadAttention, Flatten, Input

# Define checkpoint address and data columns
ckpt_addr = 'transformer_time'
keys = ['Tree species', 'Elevation', 'Temperature', 'Rainfall', 'Latitude ',' Longitude',
        'Slope gradient', 'Canopy density', 'Soil thickness', 'Humus thickness',
        'Slope direction', 'Slope position', 'DBH6', 'DBH7', 'DBH8']

# Transformer block definition
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [Dense(ff_dim, activation='relu'), Dense(embed_dim)]  # Output dimension matches input
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

# Define the model structure
class ForestTransformer(Model):
    def __init__(self):
        super(ForestTransformer, self).__init__()
        self.f1 = Dense(64, activation='relu', kernel_regularizer='l2')
        self.f2 = Dense(128, activation='relu', kernel_regularizer='l2')
        self.transformer = TransformerBlock(embed_dim=3, num_heads=2, ff_dim=32)  # embed_dim=3 for time series input
        self.flatten = Flatten()
        self.f3 = Dense(256, activation='relu', kernel_regularizer='l2')
        self.f4 = Dense(64, activation='relu', kernel_regularizer='l2')
        self.f5 = Dense(1, activation='sigmoid', kernel_regularizer='l2')

    def call(self, x):
        static_data = x[:, :-3]  # Static data part
        time_series_data = x[:, -3:]  # Time series part
        time_series_data = tf.expand_dims(time_series_data, axis=1)  # [batch_size, 1, 3]

        transformer_output = self.transformer(time_series_data)
        combined_features = tf.concat([static_data, self.flatten(transformer_output)], axis=-1)

        x = self.f3(combined_features)
        x = self.f4(x)
        y = self.f5(x)
        y = 20 * y  # Scale output
        return y

# Set GPU devices for training
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load and preprocess the data
df = pd.read_csv('../new_all_clear_data.csv', encoding='GB2312')

# Convert chest diameter columns to meters
df['DBH8'] = df['DBH8'] / 100
df['DBH7'] = df['DBH7'] / 100
df['DBH6'] = df['DBH6'] / 100

# Define target variable (rate)
rate = df['(9-8)']
y = np.array(rate)

# Define feature matrix
x = np.array(df[keys])

# Split data into training (80%) and testing (20%) sets
train_size = int(0.8 * len(x))
x_train = x[:train_size]
y_train = y[:train_size]
x_test = x[train_size:]
y_test = y[train_size:]

# Convert data to tensors
x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)
y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)

# Initialize and compile the model
model = ForestTransformer()
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='mse',
              metrics=['mean_squared_error', 'mean_absolute_error'])

# Define checkpoint path
checkpoint_save_path_num = f"./ckpt/{ckpt_addr}_rate.ckpt"
if os.path.exists(checkpoint_save_path_num + '.index'):
    print('-------------Loading the model-----------------')
    model.load_weights(checkpoint_save_path_num)

# Callback for saving the best model
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path_num,
                                                 save_weights_only=True,
                                                 monitor='val_loss',
                                                 save_best_only=True,
                                                 verbose=1)

# Custom callback for printing metrics after each epoch
class PrintMetricsCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        print(f"Epoch {epoch + 1}:")
        print(f"  Training - Loss: {logs['loss']:.4f}, MSE: {logs['mean_squared_error']:.4f}, MAE: {logs['mean_absolute_error']:.4f}")

        if 'val_loss' in logs:
            print(f"  Validation - Loss: {logs['val_loss']:.4f}, MSE: {logs['val_mean_squared_error']:.4f}, MAE: {logs['val_mean_absolute_error']:.4f}")

# Early stopping callback to prevent overfitting
early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=100,  # Stop after 100 epochs without improvement
    verbose=1,
    restore_best_weights=True  # Restore best weights after stopping
)

# Train the model with callbacks
history = model.fit(x_train, y_train, batch_size=512, shuffle=True, epochs=1000,
                    validation_data=(x_test, y_test), validation_freq=1,
                    callbacks=[cp_callback, PrintMetricsCallback(), early_stopping_callback], verbose=1)
