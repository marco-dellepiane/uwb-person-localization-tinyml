import os
import glob
import json
import re
import subprocess
import numpy as np
from scipy.optimize import linear_sum_assignment

# ==============================================================================
# CONFIGURAZIONE
# ==============================================================================
DATASET_DIR = "/home/marco/Desktop/test_project_edge_ai/other_files/dataset/data"
CODE_PY_PATH = "/home/marco/Desktop/test_project_edge_ai/submission/code.py" # Assicurati che questo sia il percorso corretto al tuo code.py

OUT_GT_DIR = "/home/marco/Desktop/test_project_edge_ai/evaluation/gt_files"
OUT_PRED_DIR = "/home/marco/Desktop/test_project_edge_ai/evaluation/pred_files"

MATCH_THRESHOLD = 1.0   

# split 1
VAL_INDICES = [23, 20, 3, 15, 7, 11] 
TRAIN_INDICES = [22, 16, 17, 18, 19, 21, 0, 1, 2, 4, 12, 13, 14, 5, 6, 8, 9, 10]

# Split2 (80% train e 22% val), stesso cocetto di split 1 "STRESS TEST sul MULTIPATH"
#VAL_INDICES = [23, 20, 0, 13, 9] 
#TRAIN_INDICES = [22, 16, 17, 18, 19, 21, 1, 2, 3, 4, 12, 14, 15, 5, 6, 7, 8, 10, 11]

os.makedirs(OUT_GT_DIR, exist_ok=True)
os.makedirs(OUT_PRED_DIR, exist_ok=True)

# ==============================================================================
# FUNZIONI DI VALUTAZIONE (Hungarian Matching)
# ==============================================================================
def _cost_matrix(gt: list, pred: list) -> np.ndarray:
    gt_arr   = np.array(gt,   dtype=np.float64)
    pred_arr = np.array(pred, dtype=np.float64)
    diff = gt_arr[:, np.newaxis, :] - pred_arr[np.newaxis, :, :]
    return np.sqrt((diff ** 2).sum(axis=-1))

def match_hungarian(gt: list, pred: list, threshold: float):
    n_gt, n_pred = len(gt), len(pred)
    if n_gt == 0 and n_pred == 0: return [], 0, 0
    if n_gt == 0: return [], 0, n_pred
    if n_pred == 0: return [], n_gt, 0
    cost = _cost_matrix(gt, pred)
    row_i, col_i = linear_sum_assignment(cost)
    matched_gt, matched_pred, distances = set(), set(), []
    for r, c in zip(row_i, col_i):
        d = float(cost[r, c])
        if d <= threshold:
            distances.append(d)
            matched_gt.add(r)
            matched_pred.add(c)
    return distances, n_gt - len(matched_gt), n_pred - len(matched_pred)

def load_jsonl(path):
    records = {}
    with open(path) as fh:
        for raw_line in fh:
            if not raw_line.strip(): continue
            obj = json.loads(raw_line)
            records[int(obj["frame"])] = [[float(xy[0]), float(xy[1])] for xy in obj.get("localizations", [])]
    return records

def generate_ground_truth(npz_path, gt_path):
    data = np.load(npz_path)
    people_xy = data['people_xy']
    people_mask = data['people_mask']
    T = people_xy.shape[0]
    with open(gt_path, "w") as f_gt:
        for t in range(T):
            gt_locs = []
            for i in range(4):
                if people_mask[t, i]:
                    gt_locs.append([float(people_xy[t, i, 0]), float(people_xy[t, i, 1])])
            f_gt.write(json.dumps({"frame": t, "localizations": gt_locs}) + "\n")

def evaluate_single_window(gt_path, pred_path, window_name, split_label):
    gt_all = load_jsonl(gt_path)
    pred_all = load_jsonl(pred_path)
    frames = sorted(set(gt_all) & set(pred_all))
    
    total_tp, total_fp, total_fn = 0, 0, 0
    all_tp_distances = []

    for frame in frames:
        gt, pred = gt_all[frame], pred_all[frame]
        dists, n_fn, n_fp = match_hungarian(gt, pred, MATCH_THRESHOLD)
        total_tp += len(dists)
        total_fp += n_fp
        total_fn += n_fn
        all_tp_distances.extend(dists)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall    = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    if all_tp_distances:
        d = np.array(all_tp_distances, dtype=np.float64)
        rmse = float(np.sqrt(np.mean(d ** 2)))
        mae = float(np.mean(d))
    else:
        rmse, mae = 0.0, 0.0

    print("=" * 65)
    print(f" 📂 FINESTRA: {window_name} | SET: {split_label} | {len(frames)} frames")
    print("=" * 65)
    if total_tp == 0 and total_fp == 0 and total_fn == 0:
        print("  🎯 Risultato: BACKGROUND PERFETTO (Nessuna persona presente/predetta).")
    else:
        print(f"  F1 Score   : {f1:.4f}")
        print(f"  Precision  : {precision:.4f}  |  Recall: {recall:.4f}")
        print(f"  TP: {total_tp} | FP: {total_fp} | FN: {total_fn}")
        if total_tp > 0:
            print(f"  Errore MAE : {mae:.4f} m   |  RMSE: {rmse:.4f} m")
    print("\n")

# ==============================================================================
# MOTORE PRINCIPALE (Usa il tuo code.py come scatola nera)
# ==============================================================================
if __name__ == "__main__":
    all_files = sorted(glob.glob(os.path.join(DATASET_DIR, "*.npz")))

    print("🚀 Inizio valutazione tramite invocazione di code.py...\n")
    
    for npz_file in all_files:
        filename = os.path.basename(npz_file)
        base_name = filename.replace('.npz', '')
        
        match = re.search(r'window_(\d+)', filename)
        if match:
            idx = int(match.group(1))
            split_label = "VALIDATION" if idx in VAL_INDICES else "TRAINING" if idx in TRAIN_INDICES else "SCONOSCIUTO"
        else:
            split_label = "N/A"

        gt_p = os.path.join(OUT_GT_DIR, f"{base_name}_gt.jsonl")
        pred_p = os.path.join(OUT_PRED_DIR, f"{base_name}_pred.jsonl")
        
        # 1. Genera la Ground Truth
        generate_ground_truth(npz_file, gt_p)
        
        # 2. ESEGUE IL TUO CODE.PY (Come farebbe il prof da terminale)
        # Comando: python code.py --input-path input.npz --output-path output.jsonl
        try:
            subprocess.run(["python", CODE_PY_PATH, 
                            "--input-path", npz_file, 
                            "--output-path", pred_p], check=True)
        except subprocess.CalledProcessError:
            print(f"❌ ERRORE CRITICO: Il tuo code.py è andato in crash sul file {filename}!")
            continue
        
        # 3. Valuta e stampa a schermo
        evaluate_single_window(gt_p, pred_p, base_name, split_label)
        
    print("✅ Analisi completata! Il tuo code.py ha processato tutti i file con successo.")