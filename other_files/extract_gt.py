import numpy as np
import json
import os

# Percorso esatto del tuo file
npz_path = "/home/marco/Desktop/test_project_edge_ai/other_files/dataset/data/window_000015.npz"
out_path = "gt_15.jsonl"

print(f"Estrazione Ground Truth da {npz_path}...")
data = np.load(npz_path)
people_xy = data['people_xy']
people_mask = data['people_mask']
T = people_xy.shape[0]

with open(out_path, "w") as f:
    for t in range(T):
        localizations = []
        for i in range(4):
            # Se la persona è valida in questo frame, salviamo la sua posizione
            if people_mask[t, i]:
                localizations.append([float(people_xy[t, i, 0]), float(people_xy[t, i, 1])])
        
        f.write(json.dumps({"frame": t, "localizations": localizations}) + "\n")

print(f"Fatto! Ground truth salvato in {out_path}")