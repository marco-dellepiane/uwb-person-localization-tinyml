import json
import matplotlib.pyplot as plt
import numpy as np

# --- CONFIGURAZIONE ---
# Usiamo la Window 19 (Serpentine walk 1 subject)
GT_FILE = "/home/marco/Desktop/test_project_edge_ai/evaluation/gt_files/window_000019_gt.jsonl"
PRED_FILE = "/home/marco/Desktop/test_project_edge_ai/evaluation/pred_files/window_000019_pred.jsonl"

ROOM_WIDTH = 4.8  # metri
ROOM_HEIGHT = 7.2 # metri

def extract_trajectory(jsonl_path):
    x_coords = []
    y_coords = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            locs = data.get("localizations", [])
            if len(locs) > 0:
                # Prendiamo solo la prima persona (ideale per le window a 1 soggetto)
                x_coords.append(locs[0][0])
                y_coords.append(locs[0][1])
    return x_coords, y_coords

# 1. Estrazione Dati
gt_x, gt_y = extract_trajectory(GT_FILE)
pred_x, pred_y = extract_trajectory(PRED_FILE)

# 2. Setup della Figura
fig, ax = plt.subplots(figsize=(6, 9)) # Proporzione 4.8 x 7.2
ax.set_xlim(-0.5, ROOM_WIDTH + 0.5)
ax.set_ylim(-0.5, ROOM_HEIGHT + 0.5)
ax.set_aspect('equal') # Mantiene le proporzioni reali della stanza
ax.grid(True, linestyle='--', alpha=0.6)

# Disegna il perimetro della stanza
stanza = plt.Rectangle((0, 0), ROOM_WIDTH, ROOM_HEIGHT, 
                       linewidth=2, edgecolor='black', facecolor='none')
ax.add_patch(stanza)

# 3. Posizionamento approssimativo dei 6 Radar (Basato sulla documentazione UWB)
# Adattali se hai le coordinate esatte dei radar TSRR250
radar_positions = [
    (ROOM_WIDTH/2, 0),             # Radar 1 (Bottom Center)
    (ROOM_WIDTH, ROOM_HEIGHT*0.25),# Radar 2 (Right Bottom)
    (ROOM_WIDTH, ROOM_HEIGHT*0.75),# Radar 3 (Right Top)
    (ROOM_WIDTH/2, ROOM_HEIGHT),   # Radar 4 (Top Center)
    (0, ROOM_HEIGHT*0.75),         # Radar 5 (Left Top)
    (0, ROOM_HEIGHT*0.25)          # Radar 6 (Left Bottom)
]
rx, ry = zip(*radar_positions)
ax.scatter(rx, ry, c='green', s=100, marker='o', label='UWB Radars', zorder=5)

# 4. Tracciamento Traiettorie
# Ground Truth: Linea tratteggiata spessa grigio scuro
ax.plot(gt_x, gt_y, color='dimgray', linestyle='--', linewidth=2.5, label='Ground Truth', zorder=3)

# Predizione: Linea continua blu
ax.plot(pred_x, pred_y, color='royalblue', linestyle='-', linewidth=2, label='TFLite INT8 Estimate', zorder=4)

# Abbellimenti
ax.set_title('Trajectory Tracking Fidelity (Window 19 - Serpentine)', fontsize=14, fontweight='bold')
ax.set_xlabel('X (Metres)', fontsize=12)
ax.set_ylabel('Y (Metres)', fontsize=12)
ax.legend(loc='upper right', framealpha=1.0)

plt.tight_layout()
plt.savefig("trajectory_plot_report.png", dpi=300)
plt.show()