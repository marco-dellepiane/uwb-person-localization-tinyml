import argparse
import numpy as np
import tensorflow as tf
import json
import sys
import os

ALPHA = 0.02 
GLOBAL_MAX = 203.10

def process_file(input_path, output_jsonl_path, tflite_model_path="model.tflite"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_full_path = os.path.join(script_dir, tflite_model_path)
    
    interpreter = tf.lite.Interpreter(
        model_path=model_full_path,
    )
    interpreter.allocate_tensors()
    
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
        bg = ALPHA * mag[t] + (1 - ALPHA) * bg
        decluttered = np.abs(mag[t] - bg)
        
        normalized = np.clip(decluttered, 0, GLOBAL_MAX) / GLOBAL_MAX
        input_tensor_float = np.expand_dims(normalized.astype(np.float32), axis=0)

        input_tensor_quant = np.round(input_tensor_float / input_scale) + input_zp
        input_tensor_quant = np.clip(input_tensor_quant, -128, 127).astype(np.int8)

        interpreter.set_tensor(input_details['index'], input_tensor_quant)
        interpreter.invoke()
        
        
        preds_quant = interpreter.get_tensor(output_details['index'])[0]
        
        preds_float = (preds_quant.astype(np.float32) - output_zp) * output_scale

        p_coords = preds_float[:8].reshape(4, 2) # first 8 are (X,Y)
        p_mask = preds_float[8:]                 # The last 4 are the Probabilities

        localizations = []
        for i in range(4):
            # Threshold to determine if the person exists
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
    parser = argparse.ArgumentParser(description="Run inference on UWB radar data.")
    parser.add_argument("--input-path", required=True, help="Path to the input .npy file")
    parser.add_argument("--output-path", required=True, help="Path to the output .jsonl file")
    args = parser.parse_args()
    
    # Esegue l'inferenza chiamando tassativamente 'model.tflite'
    process_file(args.input_path, args.output_path, tflite_model_path="model.tflite")