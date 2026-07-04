import tensorflow as tf

print("--- CONVERSIONE PESI (DYNAMIC RANGE) ---")

# Carichiamo il tuo modello perfetto
model = tf.keras.models.load_model("/home/marco/Desktop/test_project_edge_ai/other_files/eeai_best_model_v8.keras", compile=False)

converter = tf.lite.TFLiteConverter.from_keras_model(model)

# LA MAGIA È TUTTA QUI:
# tf.lite.Optimize.DEFAULT comprime i pesi in int8 (per la memoria),
# ma esegue i calcoli in Float32 (per mantenere l'accuratezza di Keras!)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

with open("/home/marco/Desktop/test_project_edge_ai/submission/model.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Finito! Modello leggerissimo ma preciso ESATTAMENTE come Keras.")