import numpy as np
import tensorflow as tf
import json
import sys
import os

ALPHA = 0.02 
#golden split 1
GLOBAL_MAX = 203.93 
# split 2
#GLOBAL_MAX = 199.43

def process_file(input_path, output_jsonl_path, tflite_model_path="golden_model_split1.tflite"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_full_path = os.path.join(script_dir, tflite_model_path)
    
    interpreter = tf.lite.Interpreter(
        model_path=model_full_path,
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
    )
    interpreter.allocate_tensors()
    
    # In questo modello C'È UN SOLO INPUT E UN SOLO OUTPUT!
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    input_scale, input_zp = input_details['quantization']
    output_scale, output_zp = output_details['quantization']

    data = np.load(input_path)
    raw_iq = data['radar_cir_iq'] if input_path.endswith('.npz') else data
    T = raw_iq.shape[0]

    mag = np.sqrt(raw_iq[..., 0]**2 + raw_iq[..., 1]**2).reshape(T, 1, 120, 18)
    bg = np.copy(mag[0])

    results = []

    for t in range(T):
        # 1. Pre-Processing e Normalizzazione (ESATTAMENTE COME NEL TRAINING)
        bg = ALPHA * mag[t] + (1 - ALPHA) * bg
        decluttered = np.abs(mag[t] - bg)
        
        normalized = np.clip(decluttered, 0, GLOBAL_MAX) / GLOBAL_MAX
        input_tensor_float = np.expand_dims(normalized.astype(np.float32), axis=0)

        # 2. QUANTIZZAZIONE DELL'INPUT
        input_tensor_quant = np.round(input_tensor_float / input_scale) + input_zp
        input_tensor_quant = np.clip(input_tensor_quant, -128, 127).astype(np.int8)

        # 3. Inferenza
        interpreter.set_tensor(input_details['index'], input_tensor_quant)
        interpreter.invoke()
        
        # 4. DEQUANTIZZAZIONE DELL'UNICO OUTPUT
        preds_quant = interpreter.get_tensor(output_details['index'])[0]
        
        # Riportiamo tutto a float (array di 12 numeri)
        preds_float = (preds_quant.astype(np.float32) - output_zp) * output_scale

        # 5. SPLIT E POST-PROCESSING (Separazione post-inferenza)
        p_coords = preds_float[:8].reshape(4, 2) # I primi 8 sono (X,Y)
        p_mask = preds_float[8:]                 # Gli ultimi 4 sono le Probabilità

        localizations = []
        for i in range(4):
            # Soglia per capire se la persona esiste
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