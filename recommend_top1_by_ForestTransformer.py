import pandas as pd
import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Embedding

# Load the data file
input_file = './top4.csv'

ckpt_addr = 'transformer_time'

# Define the new keys for ForestTransformer
keys = ['Tree species', 'Elevation', 'Temperature', 'Rainfall', 'Latitude ',' Longitude',
        'Slope gradient', 'Canopy density', 'Soil thickness', 'Humus thickness',
        'Slope direction', 'Slope position', 'DBH7', 'DBH8', 'DBH9']

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

# Load normalization min-max values
df_min_max = pd.read_csv('../yulin_min_max.csv', encoding='gb18030')
df_min_max = df_min_max.set_index(['keys'])
species_max = df_min_max.loc['Tree_Species_Code', 'max']
species_min = df_min_max.loc['Tree_Species_Code', 'min']

def normalize(x, keys):
    df_min_max = pd.read_csv('../yulin_min_max.csv', encoding='gb18030')
    df_min_max = df_min_max.set_index(['keys'])
    max_vals = [df_min_max['max'][key] for key in keys]
    min_vals = [df_min_max['min'][key] for key in keys]

    normalized_data = []
    for row in x:
        normalized_row = [(value - min_vals[i]) / (max_vals[i] - min_vals[i]) for i, value in enumerate(row)]
        normalized_data.append(normalized_row)
    return np.array(normalized_data)

# Read the file
df = pd.read_csv(input_file, encoding='GB2312', sep='\t')

# Extract data
x = df[[f"Original_{key}" for key in keys]].values

x_normalized = normalize(x, keys)

x_predict = tf.convert_to_tensor(x_normalized, dtype=tf.float32)

# Define model
num_species = int(species_max - species_min + 1)
embed_dim = 8  # Embedding dimension
model = ForestTransformer()

model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mean_squared_error')

checkpoint_save_path_num = "./ckpt/" + ckpt_addr + "_shengzhang.ckpt"
if os.path.exists(checkpoint_save_path_num + '.index'):
    print('-------------Load the model-----------------')
    model.load_weights(checkpoint_save_path_num)

predictions = model.predict(x_predict)

# Add predictions to the dataframe and save as CSV
df['Predictions'] = predictions
df.to_csv('yuan_shuzhong.csv', index=False, encoding='utf-8-sig')
print("Predictions saved to 'yuan_shuzhong.csv'")

# Prediction logic
species_results = []
for index, row in df.iterrows():
    original_prediction = predictions[index]
    species_changes = [row[f'Top_{i}'] for i in range(1, 5)]

    predictions_list = []
    for change in species_changes:
        row_copy = x_normalized[index].copy()
        row_copy[0] = (change - species_min) / (species_max - species_min)  # Normalization
        row_copy = tf.convert_to_tensor(row_copy, dtype=tf.float32)
        row_copy = tf.reshape(row_copy, (1, -1))

        predicted_volume = model.predict(row_copy)[0]
        predictions_list.append({'Change': change, 'Predicted_Volume': predicted_volume[0]})

    predictions_list = sorted(predictions_list, key=lambda x: x['Predicted_Volume'], reverse=True)[:3]

    top_changes = {
        'Original_Species': row,
        'Original_Predicted_Volume': original_prediction[0],
        'Top_Changes': [(item['Change'], item['Predicted_Volume']) for item in predictions_list]
    }
    species_results.append(top_changes)

density_results_df = pd.DataFrame([
    {
        'Species_Code': result['Top_Changes'][0][0],
        'Volume_1': result['Top_Changes'][0][1],
        'Latitude': result['Original_Species']['Original_Latitude'],
        'Longitude': result['Original_Species']['Original_Longitude']
    }
    for result in species_results
])

density_output_path = "./top1.csv"
density_results_df.to_csv(density_output_path, index=False, encoding='gb18030')