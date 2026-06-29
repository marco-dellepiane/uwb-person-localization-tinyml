import numpy as np
import tensorflow as tf
import json
import sys
import os

# ==============================================================================
# 1. PARAMETRI DI NORMALIZZAZIONE (HARDCODED)
# Sostituisci questi 0.0 e 1.0 con i valori reali stampati nel training!
# ==============================================================================
TRAIN_MEAN = 0.0  # <--- INSERISCI IL VALORE REALE
TRAIN_STD = 1.0   # <--- INSERISCI IL VALORE REALE

def process_file(input_path, output_jsonl_path, tflite_model_path="model.tflite"):
    print(f"Caricamento modello TFLite: {tflite_model_path}")
    
    # 2. Inizializzazione Interprete TFLite
    # Nota: il file model.tflite deve trovarsi nella stessa cartella di questo script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_full_path = os.path.join(script_dir, tflite_model_path)
    
    interpreter = tf.lite.Interpreter(model_path=model_full_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    # Estrazione dei parametri di scala per la quantizzazione INT8
    input_scale, input_zp = input_details['quantization']
    output_scale, output_zp = output_details['quantization']

    # 3. Caricamento Dati
    print(f"Elaborazione file: {input_path}")
    # Gestiamo sia .npy (come da specifica) sia .npz (come i file di training)
    if input_path.endswith('.npz'):
        data = np.load(input_path)
        raw_iq = data['radar_cir_iq']
    else:
        raw_iq = np.load(input_path)
        
    T = raw_iq.shape[0]

    # Calcolo magnitudo iniziale
    mag = np.sqrt(raw_iq[..., 0]**2 + raw_iq[..., 1]**2)
    mag_reshaped = mag.reshape(T, 1, 120, 18)

    # Inizializzazione buffer EMA
    bg = np.copy(mag_reshaped[0])
    alpha = 0.20

    results = []

    for t in range(T):
        # A. Preprocessing: EMA Decluttering sequenziale
        bg = alpha * mag_reshaped[t] + (1 - alpha) * bg
        decluttered = np.abs(mag_reshaped[t] - bg)

        # B. Normalizzazione Rigorosa
        normalized = (decluttered - TRAIN_MEAN) / (TRAIN_STD + 1e-7)

        # C. Quantizzazione (Float32 -> INT8)
        input_int8 = np.round(normalized / input_scale + input_zp).astype(np.int8)
        
        # D. Inferenza
        interpreter.set_tensor(input_details['index'], input_int8)
        interpreter.invoke()
        
        # E. De-quantizzazione (INT8 -> Float32)
        output_int8 = interpreter.get_tensor(output_details['index'])
        output_float32 = (output_int8.astype(np.float32) - output_zp) * output_scale

        # F. Post-processing: Estrazione coordinate e maschera
        coords = output_float32[0, :8].reshape(4, 2)
        mask_probs = output_float32[0, 8:] 

        # Generazione lista localizzazioni (solo se la rete è sicura > 50%)
        localizations = []
        for i in range(4):
            if mask_probs[i] > 0.5:
                # Arrotondiamo a 3 decimali (millimetri)
                localizations.append([round(float(coords[i, 0]), 3), round(float(coords[i, 1]), 3)])

        results.append({
            "frame": t,
            "localizations": localizations
        })

    # 4. Salvataggio Output JSONL
    print(f"Salvataggio risultati in: {output_jsonl_path}")
    with open(output_jsonl_path, 'w') as f:
        for res in results:
            f.write(json.dumps(res) + '\n')
            
    print("Elaborazione completata con successo!")

if __name__ == "__main__":
    if len(sys.argv) == 3:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        process_file(input_file, output_file)
    else:
        print("Uso corretto: python code.py <percorso_input> <percorso_output.jsonl>")