"""
Evaluation script for Voice-Based Authentication System.
Calculates Equal Error Rate (EER), AUC-ROC, and optimal decision threshold
using genuine and impostor speaker pairs from the VoxCeleb dataset.
Supports comparing Resemblyzer vs custom VoiceEmbeddingCNN side-by-side.
"""

import os
import sys
import time
import glob
import argparse
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
from scipy.optimize import brentq
from scipy.interpolate import interp1d

from ml.model import extract_embedding, extract_embedding_cnn

# Ensure immediate unbuffered output in terminal and logs
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

# ----------------- Configuration -----------------
DATASET_PATH = r"D:\vox1_test_wav\wav"
if not os.path.exists(DATASET_PATH):
    alt_path = r"D:\vox1_test_wav"
    if os.path.exists(alt_path):
        DATASET_PATH = alt_path

MAX_SPEAKERS = 20
SAMPLES_PER_SPEAKER = 6
# -------------------------------------------------


def compute_eer(y_true: np.ndarray, y_scores: np.ndarray):
    """
    Compute Equal Error Rate (EER) and the corresponding decision threshold.

    Args:
        y_true: 1D binary array (1 = genuine pair, 0 = impostor pair)
        y_scores: 1D array of similarity scores in [-1.0, 1.0]

    Returns:
        tuple (eer, eer_threshold, auc)
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr

    # Numerical approximation as fallback
    diff = np.abs(fpr - fnr)
    best_idx = np.argmin(diff)
    eer_approx = (fpr[best_idx] + fnr[best_idx]) / 2.0
    thresh_approx = thresholds[best_idx]

    try:
        # Exact interpolated intersection where FPR == FNR
        interp_tpr = interp1d(fpr, tpr)
        eer_exact = brentq(lambda x: 1.0 - x - interp_tpr(x), 0.0, 1.0)
        interp_thresh = interp1d(fpr, thresholds, bounds_error=False, fill_value="extrapolate")
        eer_threshold = float(interp_thresh(eer_exact))
        eer = float(eer_exact)
    except Exception:
        eer = float(eer_approx)
        eer_threshold = float(thresh_approx)

    auc = roc_auc_score(y_true, y_scores)
    return eer, eer_threshold, auc


def run_evaluation(extractor_fn, model_name: str, file_items: list):
    """
    Extract embeddings using extractor_fn and compute EER on pairwise comparisons.
    """
    print(f"\nEvaluating: {model_name}...")
    total_files = len(file_items)
    embeddings = []
    speaker_labels = []
    t0 = time.time()

    for idx, (wav_file, spk_idx) in enumerate(file_items):
        try:
            emb = extractor_fn(wav_file)
            embeddings.append(emb)
            speaker_labels.append(spk_idx)
        except Exception as e:
            pass

        if (idx + 1) % 30 == 0 or (idx + 1) == total_files:
            elapsed = time.time() - t0
            print(f"  [{model_name}] Processed {idx + 1}/{total_files} files ({elapsed:.1f}s, {elapsed / (idx + 1):.2f}s/file)")

    extract_time = time.time() - t0
    embeddings = np.array(embeddings, dtype=np.float32)
    speaker_labels = np.array(speaker_labels, dtype=np.int64)

    # Compute pairwise cosine similarity matrix
    sim_matrix = np.dot(embeddings, embeddings.T)

    y_true = []
    y_scores = []
    n = len(speaker_labels)

    for i in range(n):
        for j in range(i + 1, n):
            is_genuine = int(speaker_labels[i] == speaker_labels[j])
            score = float(sim_matrix[i, j])
            y_true.append(is_genuine)
            y_scores.append(score)

    y_true = np.array(y_true, dtype=np.int32)
    y_scores = np.array(y_scores, dtype=np.float32)

    n_genuine = int(np.sum(y_true == 1))
    n_impostor = int(np.sum(y_true == 0))

    genuine_scores = y_scores[y_true == 1]
    impostor_scores = y_scores[y_true == 0]

    eer, eer_threshold, auc = compute_eer(y_true, y_scores)

    return {
        "model_name": model_name,
        "eer_percent": round(eer * 100, 2),
        "eer_threshold": round(eer_threshold, 4),
        "auc_roc": round(auc, 4),
        "total_pairs": len(y_true),
        "genuine_pairs": n_genuine,
        "impostor_pairs": n_impostor,
        "mean_genuine": round(float(np.mean(genuine_scores)), 4),
        "std_genuine": round(float(np.std(genuine_scores)), 4),
        "mean_impostor": round(float(np.mean(impostor_scores)), 4),
        "std_impostor": round(float(np.std(impostor_scores)), 4),
        "separation_margin": round(float(np.mean(genuine_scores) - np.mean(impostor_scores)), 4),
        "total_extract_time": round(extract_time, 2),
        "ms_per_file": round((extract_time / max(1, total_files)) * 1000, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate speaker verification models.")
    parser.add_argument("--model", choices=["all", "cnn", "resemblyzer"], default="all",
                        help="Model to evaluate (default: all for side-by-side comparison)")
    parser.add_argument("--speakers", type=int, default=MAX_SPEAKERS, help="Number of speakers")
    parser.add_argument("--samples", type=int, default=SAMPLES_PER_SPEAKER, help="Samples per speaker")
    args = parser.parse_args()

    print("=" * 72)
    print(" Voice Biometric Authentication - Side-by-Side Model Evaluation")
    print(f" Dataset Path: {DATASET_PATH}")
    print(f" Configuration: {args.speakers} speakers, {args.samples} samples/speaker")
    print("=" * 72)

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset path not found: {DATASET_PATH}")

    speaker_dirs = sorted([
        d for d in os.listdir(DATASET_PATH)
        if os.path.isdir(os.path.join(DATASET_PATH, d))
    ])[:args.speakers]

    file_items = []
    for spk_idx, spk_id in enumerate(speaker_dirs):
        spk_path = os.path.join(DATASET_PATH, spk_id)
        wav_files = sorted(glob.glob(os.path.join(spk_path, "**", "*.wav"), recursive=True))
        selected_wavs = wav_files[:args.samples]
        for w in selected_wavs:
            file_items.append((w, spk_idx))

    print(f"Gathered {len(file_items)} audio recordings across {len(speaker_dirs)} speakers.")

    results = []

    # 1. Resemblyzer Evaluation
    if args.model in ["all", "resemblyzer"]:
        res_emb = run_evaluation(extract_embedding, "Resemblyzer (Pretrained d-vector)", file_items)
        results.append(res_emb)

    # 2. Custom CNN Evaluation
    if args.model in ["all", "cnn"]:
        res_cnn = run_evaluation(extract_embedding_cnn, "VoiceEmbeddingCNN (Trained on VoxCeleb)", file_items)
        results.append(res_cnn)

    # Side-by-Side Comparison Output
    print("\n" + "=" * 76)
    print("               BIOMETRIC MODEL COMPARISON (SIDE-BY-SIDE)                 ")
    print("=" * 76)
    header = f"{'Metric':<32} | {'Resemblyzer (d-vector)':<20} | {'VoiceEmbeddingCNN':<20}"
    print(header)
    print("-" * 76)

    if len(results) == 2:
        r_res, r_cnn = results[0], results[1]
        print(f"{'Equal Error Rate (EER)':<32} | {r_res['eer_percent']:>18.2f}% | {r_cnn['eer_percent']:>18.2f}%")
        print(f"{'Optimal EER Threshold':<32} | {r_res['eer_threshold']:>20.4f} | {r_cnn['eer_threshold']:>20.4f}")
        print(f"{'Area Under ROC (AUC-ROC)':<32} | {r_res['auc_roc']:>20.4f} | {r_cnn['auc_roc']:>20.4f}")
        print(f"{'Mean Genuine Score':<32} | {r_res['mean_genuine']:>20.4f} | {r_cnn['mean_genuine']:>20.4f}")
        print(f"{'Mean Impostor Score':<32} | {r_res['mean_impostor']:>20.4f} | {r_cnn['mean_impostor']:>20.4f}")
        print(f"{'Separation Margin':<32} | {r_res['separation_margin']:>19.4f} | {r_cnn['separation_margin']:>19.4f}")
        print(f"{'Embedding Dimension':<32} | {'256':>20} | {'128':>20}")
        print(f"{'Total Comparisons':<32} | {r_res['total_pairs']:>20,d} | {r_cnn['total_pairs']:>20,d}")
        print(f"{'Inference Speed (per sample)':<32} | {r_res['ms_per_file']:>18.1f}ms | {r_cnn['ms_per_file']:>18.1f}ms")
    else:
        for r in results:
            print(f"Model: {r['model_name']}")
            print(f"  EER: {r['eer_percent']}% at threshold {r['eer_threshold']}")
            print(f"  AUC-ROC: {r['auc_roc']}")
            print(f"  Margin: {r['separation_margin']}")
    print("=" * 76)

    return results


if __name__ == "__main__":
    main()
