import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import cv2  # <--- for visualization

from args import get_args
from dataset import LungXrayDataset
from models import MyModel

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)


def evaluate():
    args = get_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ---------- 1. Load test CSV ----------
    test_csv = os.path.join(args.csv_dir, "test.csv")
    if not os.path.exists(test_csv):
        raise FileNotFoundError(f"Could not find test CSV at {test_csv}")

    test_df = pd.read_csv(test_csv)
    print("Test samples:", len(test_df))

    # Dataset & DataLoader
    test_dataset = LungXrayDataset(test_df)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,      # safer on Windows
        pin_memory=False
    )

    num_classes = args.num_classes  # 2
    num_samples = len(test_dataset)

    # ---------- 2. Collect predictions from each fold ----------
    prob_sum = np.zeros((num_samples, num_classes), dtype=np.float32)
    all_targets = np.zeros(num_samples, dtype=np.int64)

    n_folds_used = 0

    for fold in range(1, 6):
        model_path = os.path.join(args.out_dir, f"best_model_fold_{fold}.pth")
        if not os.path.exists(model_path):
            print(f"[Warning] Model for fold {fold} not found at {model_path}, skipping.")
            continue

        print(f"Loading model from: {model_path}")
        model = MyModel(backbone=args.backbone, num_classes=num_classes)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        n_folds_used += 1

        idx = 0
        with torch.no_grad():
            for batch in test_loader:
                inputs = batch["image"].to(device)
                targets = batch["label"].cpu().numpy()  # shape (B,)

                logits = model(inputs)
                probs = torch.softmax(logits, dim=1).cpu().numpy()  # (B,2)

                bsz = probs.shape[0]
                prob_sum[idx:idx + bsz] += probs

                # Save targets once (same order every time because shuffle=False)
                if fold == 1:
                    all_targets[idx:idx + bsz] = targets

                idx += bsz

    if n_folds_used == 0:
        raise RuntimeError("No fold models were found to evaluate on test set.")

    # ---------- 3. Average probabilities across folds ----------
    prob_avg = prob_sum / n_folds_used
    preds = np.argmax(prob_avg, axis=1)  # predicted class 0/1

    # ---------- 4. Compute metrics ----------
    acc = accuracy_score(all_targets, preds)
    bal_acc = balanced_accuracy_score(all_targets, preds)

    # For ROC AUC and AP in binary case, use probability of class 1 (Pneumonia)
    y_true = all_targets
    y_prob_pos = prob_avg[:, 1]

    try:
        roc_auc = roc_auc_score(y_true, y_prob_pos)
        avg_prec = average_precision_score(y_true, y_prob_pos)
    except ValueError:
        roc_auc, avg_prec = 0.0, 0.0

    cm = confusion_matrix(all_targets, preds)

    print("\n===== Test Set Results (Ensembled over folds) =====")
    print(f"Accuracy:           {acc:.4f}")
    print(f"Balanced Accuracy:  {bal_acc:.4f}")
    print(f"ROC AUC:            {roc_auc:.4f}")
    print(f"Average Precision:  {avg_prec:.4f}")
    print("Confusion Matrix (rows = true, cols = pred):")
    print(cm)
    print("\nClassification report:")
    print(classification_report(all_targets, preds, target_names=["Normal", "Pneumonia"]))

    # ---------- 5. Save predictions to CSV ----------
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    out_df = test_df.copy()
    out_df["PredLabel"] = preds
    out_df["Prob_Normal"] = prob_avg[:, 0]
    out_df["Prob_Pneumonia"] = prob_avg[:, 1]

    out_csv_path = os.path.join(out_dir, "test_predictions.csv")
    out_df.to_csv(out_csv_path, index=False)
    print(f"\n✔ Test predictions saved to: {out_csv_path}")

    # ---------- 6. Save images with overlay (Pred / True) ----------
    vis_dir = os.path.join(out_dir, "predicted_images")
    os.makedirs(vis_dir, exist_ok=True)

    id2name = {0: "Normal", 1: "Pneumonia"}

    print(f"Saving visualized predictions to: {vis_dir}")

    for i, row in out_df.iterrows():
        img_path = row["Path"]
        true_label = int(row["Label"])
        pred_label = int(row["PredLabel"])
        prob_pneu = float(row["Prob_Pneumonia"])

        true_name = id2name.get(true_label, str(true_label))
        pred_name = id2name.get(pred_label, str(pred_label))

        # Read original image (grayscale or color, both fine)
        img = cv2.imread(img_path)
        if img is None:
            print(f"[Warning] Could not read image: {img_path}")
            continue

        # Decide color: green if correct, red if wrong
        correct = (true_label == pred_label)
        color = (0, 255, 0) if correct else (0, 0, 255)  # BGR

        # Put text on image
        text_pred = f"Pred: {pred_name} ({prob_pneu:.2f})"
        text_true = f"True: {true_name}"

        cv2.putText(img, text_pred, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, color, 2, cv2.LINE_AA)
        cv2.putText(img, text_true, (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2, cv2.LINE_AA)

        # Save under same filename in predicted_images/
        filename = os.path.basename(img_path)
        out_img_path = os.path.join(vis_dir, filename)
        cv2.imwrite(out_img_path, img)

    print("✔ Visualization images saved.")


if __name__ == "__main__":
    evaluate()
