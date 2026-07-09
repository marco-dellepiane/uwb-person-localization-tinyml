import tensorflow as tf
import numpy as np

# Carica il modello "Golden" addestrato
model = tf.keras.models.load_model("/home/marco/Desktop/test_project_edge_ai/other_files/golden_model.keras", compile=False)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# COSTANTI FONDAMENTALI (Devono coincidere col Training!)
ALPHA = 0.02 
# golden
GLOBAL_MAX = 203.93  
# split 2
#GLOBAL_MAX = 199.43 

def representative_data_gen():
    calibration_files = [
        "/home/marco/Desktop/test_project_edge_ai/other_files/dataset/data/window_000023.npz", # 0 soggetti
        "/home/marco/Desktop/test_project_edge_ai/other_files/dataset/data/window_000015.npz", # Misti
        "/home/marco/Desktop/test_project_edge_ai/other_files/dataset/data/window_000007.npz"  # 4 soggetti
    ]
    
    frames_per_file = 250

    for file_path in calibration_files:
        data = np.load(file_path)
        raw_iq = data['radar_cir_iq']
        
        # Gestione sicura nel caso il file abbia meno di 100 frame
        limit = min(frames_per_file, raw_iq.shape[0])
        
        mag = np.sqrt(raw_iq[:limit, ..., 0]**2 + raw_iq[:limit, ..., 1]**2).reshape(limit, 1, 120, 18)
        bg = np.copy(mag[0])
        
        for t in range(limit):
            bg = ALPHA * mag[t] + (1 - ALPHA) * bg
            decluttered = np.abs(mag[t] - bg)
            
            # NORMALIZZAZIONE OBBLIGATORIA: Come nel training!
            normalized = np.clip(decluttered, 0, GLOBAL_MAX) / GLOBAL_MAX
            
            input_tensor = np.expand_dims(normalized.astype(np.float32), axis=0)
            yield [input_tensor]

converter.representative_dataset = representative_data_gen

# Full INT8 enforcement per ESP32-S3
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

print("Inizio quantizzazione INT8... (potrebbe richiedere qualche minuto)")
tflite_model = converter.convert()

with open("golden_model_split1.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Quantizzazione Full INT8 completata con successo!")