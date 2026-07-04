import numpy as np
import tensorflow as tf

print("=== DIAGNOSTICA PURA TFLITE ===")
interpreter = tf.lite.Interpreter(model_path="submission/model.tflite")
interpreter.allocate_tensors()

in_details = interpreter.get_input_details()[0]
out_details = interpreter.get_output_details()

print(f"Numero di output trovati: {len(out_details)}\n")

for i, out in enumerate(out_details):
    print(f"OUTPUT {i}:")
    print(f"  - Forma (Shape): {out['shape']}")
    print(f"  - Scala e ZeroPoint: {out['quantization']}")

# Carichiamo 1 solo frame dalla finestra 20
data = np.load("other_files/dataset/data/window_000020.npz")
mag = np.sqrt(data['radar_cir_iq'][10:11, ..., 0]**2 + data['radar_cir_iq'][10:11, ..., 1]**2)
mag_reshaped = mag.reshape(1, 1, 120, 18)

# Normalizziamo usando i tuoi valori V8
normalized = (mag_reshaped - 13.715407) / (45.612080 + 1e-7)

# Quantizziamo
scale, zp = in_details['quantization']
q_val = np.round(normalized / scale + zp)
input_int8 = np.clip(q_val, -128, 127).astype(np.int8)

# Inferenza
interpreter.set_tensor(in_details['index'], input_int8)
interpreter.invoke()

print("\n=== VALORI DECIFRATI DAL MODELLO ===")
for i, out in enumerate(out_details):
    tensor = interpreter.get_tensor(out['index'])
    scale, zp = out['quantization']
    float_val = (tensor.astype(np.float32) - zp) * scale
    print(f"Valori reali Output {i}: {float_val.flatten()}")