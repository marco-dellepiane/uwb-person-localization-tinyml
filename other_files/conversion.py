import tensorflow as tf
import numpy as np
import os
import glob

print("--- INIZIO CONVERSIONE TFLITE INT8 (STANDALONE) ---")

# ==============================================================================
# 1. INSERISCI I VALORI CHE TI ERI SALVATO DURANTE IL TRAINING!
# ==============================================================================
TRAIN_MEAN = 10.366536  # <--- METTI IL TUO VALORE QUI
TRAIN_STD = 37.227516  # <--- METTI IL TUO VALORE QUI

# ==============================================================================
# 2. CARICAMENTO DI UN MICRO-CAMPIONE DI DATI (Solo 2 file per fare in fretta)
# ==============================================================================
print("Caricamento di un piccolo campione di dati per la calibrazione...")

# Selezioniamo le finestre tattiche: 
# Vuoto (22), Seduto (21), Caos/4 Persone (5), Misto (10)
train_indices_to_load = [22, 21, 5, 10]
tutti_i_file = glob.glob("/home/marco/Desktop/test_project_edge_ai/other_files/dataset/data/*.npz")
print(f"Trovati {len(tutti_i_file)} file .npz totali nella cartella!")
sample_files = [f for f in tutti_i_file if int(os.path.basename(f).replace("window_", "").replace(".npz", "")) in train_indices_to_load]
print(f"File selezionati per la calibrazione: {sample_files}")


X_sample_list = []
alpha = 0.20

for file_path in sample_files:
    data = np.load(file_path)
    raw_iq = data['radar_cir_iq']
    T = raw_iq.shape[0]
    
    mag = np.sqrt(raw_iq[..., 0]**2 + raw_iq[..., 1]**2)
    mag_reshaped = mag.reshape(T, 1, 120, 18)
    
    bg = np.copy(mag_reshaped[0])
    decluttered = np.zeros_like(mag_reshaped)
    
    for t in range(T):
        bg = alpha * mag_reshaped[t] + (1 - alpha) * bg
        decluttered[t] = np.abs(mag_reshaped[t] - bg)
        
    X_sample_list.append(decluttered)

# Uniamo e normalizziamo usando le statistiche del VERO training
X_sample_raw = np.concatenate(X_sample_list, axis=0).astype(np.float32)
X_calib = (X_sample_raw - TRAIN_MEAN) / (TRAIN_STD + 1e-7)
print(f"Campione pronto: {X_calib.shape[0]} frame disponibili in RAM.")

# ==============================================================================
# 3. CONVERSIONE TFLITE INT8
# ==============================================================================
# Carica il modello salvato su disco
model = tf.keras.models.load_model("/home/marco/Desktop/test_project_edge_ai/other_files/eeai_best_model_v8.keras", compile=False)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Generatore per la calibrazione INT8 (Prende 250 frame a caso dal nostro mini-campione)
def representative_dataset():
    indices = np.random.choice(X_calib.shape[0], size=250, replace=False)
    for i in indices:
        sample = X_calib[i:i+1].astype(np.float32)
        yield [sample]

converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

# Esecuzione conversione
print("Quantizzazione in corso (potrebbe richiedere un minuto)...")
tflite_quant_model = converter.convert()

tflite_path = "/home/marco/Desktop/test_project_edge_ai/submission/model.tflite"
with open(tflite_path, "wb") as f:
    f.write(tflite_quant_model)

print(f"✅ Conversione completata!")
print(f"💾 Dimensione modello TFLite: {os.path.getsize(tflite_path) / 1024:.2f} KB (Limite: 800 KB)")