import numpy as np
import tensorflow as tf

# Carichiamo entrambi i modelli
model_keras = tf.keras.models.load_model("/home/marco/Desktop/test_project_edge_ai/other_files/eeai_best_model_v8.keras", compile=False)
interpreter = tf.lite.Interpreter(model_path="/home/marco/Desktop/test_project_edge_ai/submission/model.tflite")
interpreter.allocate_tensors()

# Prendiamo un frame a caso dal dataset
data = np.load("/home/marco/Desktop/test_project_edge_ai/other_files/dataset/data/window_000020.npz")
sample = data['radar_cir_iq'][10:11] # un frame a caso
mag = np.sqrt(sample[..., 0]**2 + sample[..., 1]**2).reshape(1, 1, 120, 18)

# 1. TEST KERAS (FLOAT32)
pred_keras = model_keras.predict(mag)
print(f"Probabilità KERAS: {pred_keras[0, 8:]}")

# 2. TEST TFLITE (INT8)
input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]
# Normalizzazione grezza per il test
normalized = (mag - 10.366536) / 37.227516
input_int8 = np.round(normalized / input_details['quantization'][0] + input_details['quantization'][1]).astype(np.int8)
interpreter.set_tensor(input_details['index'], input_int8)
interpreter.invoke()
pred_tflite = interpreter.get_tensor(output_details['index'])
# De-quantizzazione
pred_tflite_float = (pred_tflite.astype(np.float32) - output_details['quantization'][1]) * output_details['quantization'][0]
print(f"Probabilità TFLITE: {pred_tflite_float[0, 8:]}")