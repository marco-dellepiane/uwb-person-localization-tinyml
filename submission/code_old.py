import numpy as np
import tensorflow as tf
import json
import sys
import os

# I valori della tua V8
TRAIN_MEAN = 13.715407
TRAIN_STD = 45.612080

def process_file(input_path, output_jsonl_path, tflite_model_path="model.tflite"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_full_path = os.path.join(script_dir, tflite_model_path)
    
    interpreter = tf.lite.Interpreter(model_path=model_full_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    # Caricamento Dati
    data = np.load(input_path)
    raw_iq = data['radar_cir_iq'] if input_path.endswith('.npz') else data
    T = raw_iq.shape[0]

    mag = np.sqrt(raw_iq[..., 0]**2 + raw_iq[..., 1]**2).reshape(T, 1, 120, 18)
    bg = np.copy(mag[0])
    alpha = 0.002

    results = []

    for t in range(T):
        # 1. EMA (Esattamente come in Keras)
        bg = alpha * mag[t] + (1 - alpha) * bg
        decluttered = np.abs(mag[t] - bg)

        # 2. Normalizzazione
        normalized = (decluttered - TRAIN_MEAN) / (TRAIN_STD + 1e-7)
        
        # 3. Preparazione Input (Float32 puro)
        input_tensor = np.expand_dims(normalized.astype(np.float32), axis=0)

        # 4. Inferenza
        interpreter.set_tensor(input_details['index'], input_tensor)
        interpreter.invoke()
        
        # L'output è già un array Float32 pulito! [1, 12]
        preds = interpreter.get_tensor(output_details['index'])[0]

        # 5. Slicing e Post-Processing (Come nel visualizzatore)
        p_coords = preds[:8].reshape(4, 2)
        p_mask = preds[8:]

        localizations = []
        for i in range(4):
            if float(p_mask[i]) >= 0.85:
                localizations.append([round(float(p_coords[i, 0]), 3), round(float(p_coords[i, 1]), 3)])

        results.append({
            "frame": t,
            "localizations": localizations
        })

    with open(output_jsonl_path, 'w') as f:
        for res in results:
            f.write(json.dumps(res) + '\n')

if __name__ == "__main__":
    if len(sys.argv) == 3:
        process_file(sys.argv[1], sys.argv[2])