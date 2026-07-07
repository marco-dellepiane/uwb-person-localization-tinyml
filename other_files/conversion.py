import tensorflow as tf
import numpy as np

# Carica il modello addestrato sui dati GREZZI
model = tf.keras.models.load_model("/home/marco/Desktop/test_project_edge_ai/other_files/best_model_toscano1.keras", compile=False)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# L'alpha DEVE essere quello usato nel training!
ALPHA = 0.02 

def representative_data_gen():
    calibration_files = [
        "/home/marco/Desktop/test_project_edge_ai/other_files/dataset/data/window_000023.npz", # 0 soggetti
        "/home/marco/Desktop/test_project_edge_ai/other_files/dataset/data/window_000015.npz", # Misti
        "/home/marco/Desktop/test_project_edge_ai/other_files/dataset/data/window_000007.npz"  # 4 soggetti
    ]
    
    frames_per_file = 100

    for file_path in calibration_files:
        data = np.load(file_path)
        raw_iq = data['radar_cir_iq']
        
        mag = np.sqrt(raw_iq[:frames_per_file, ..., 0]**2 + raw_iq[:frames_per_file, ..., 1]**2).reshape(frames_per_file, 1, 120, 18)
        bg = np.copy(mag[0])
        
        for t in range(frames_per_file):
            bg = ALPHA * mag[t] + (1 - ALPHA) * bg
            decluttered = np.abs(mag[t] - bg)
            
            # NESSUNA NORMALIZZAZIONE: passiamo il decluttered grezzo!
            input_tensor = np.expand_dims(decluttered.astype(np.float32), axis=0)
            yield [input_tensor]

converter.representative_dataset = representative_data_gen

# Full INT8 enforcement
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()
with open("submission/model.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Quantizzazione Full INT8 (sui dati grezzi) completata!")