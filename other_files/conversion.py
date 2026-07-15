import os
import glob
import numpy as np
import tensorflow as tf

ALPHA = 0.02


GLOBAL_MAX = 203.10


print("1. Caricamento del modello Keras 'Golden'...")
model = tf.keras.models.load_model("/home/marco/Desktop/test_project_edge_ai/other_files/split6.keras", compile=False)

print("2. Forzatura della Batch Size a 1 (Fix per XNNPACK)...")
# Questo step previene il crash sul PC del professore
run_model = tf.function(lambda x: model(x))
concrete_func = run_model.get_concrete_function(
    tf.TensorSpec([1, 1, 120, 18], tf.float32)
)

print("3. Preparazione dei dati di calibrazione (Sulle tue 3 finestre)...")
def representative_data_gen():
    calibration_files = [
        "/home/marco/Desktop/test_project_edge_ai/other_files/dataset/data/window_000023.npz", # 0 soggetti
        "/home/marco/Desktop/test_project_edge_ai/other_files/dataset/data/window_000015.npz", # Misti
        "/home/marco/Desktop/test_project_edge_ai/other_files/dataset/data/window_000007.npz"  # 4 soggetti
    ]
    
    frames_per_file = 400

    for file_path in calibration_files:
        data = np.load(file_path)
        raw_iq = data['radar_cir_iq']
        
        limit = min(frames_per_file, raw_iq.shape[0])
        mag = np.sqrt(raw_iq[:limit, ..., 0]**2 + raw_iq[:limit, ..., 1]**2).reshape(limit, 1, 120, 18)
        bg = np.copy(mag[0])
        
        for t in range(limit):
            bg = ALPHA * mag[t] + (1 - ALPHA) * bg
            decluttered = np.abs(mag[t] - bg)
            
            normalized = np.clip(decluttered, 0, GLOBAL_MAX) / GLOBAL_MAX
            input_tensor = np.expand_dims(normalized.astype(np.float32), axis=0)
            yield [input_tensor]

print("4. Quantizzazione INT8 in corso...")
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen

# Full INT8 enforcement + TFLITE_BUILTINS per la massima compatibilità
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()

os.makedirs("submission", exist_ok=True)
with open("submission/model.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Quantizzazione Full INT8 completata con successo! Modello pronto per il test dei vincoli.")