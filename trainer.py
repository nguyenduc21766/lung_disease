from args import get_args
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import balanced_accuracy_score, roc_auc_score, average_precision_score
from sklearn.preprocessing import label_binarize

import matplotlib.pyplot as plt
import numpy as np
import os


def train_model(model, train_loader, val_loader, fold=1, device="cpu"):
    """
    Train the model for one fold.
    Saves the best model (by validation balanced accuracy)
    and training curves to args.out_dir.
    """
    args = get_args()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    history = {
        "train_loss": [], "val_loss": [],
        "train_bal_acc": [], "val_bal_acc": [],
        "train_roc_auc": [], "val_roc_auc": [],
        "train_avg_prec": [], "val_avg_prec": [],
    }

    best_bal_acc = 0.0
    os.makedirs(args.out_dir, exist_ok=True)
    num_classes = args.num_classes  # 2 for pneumonia vs normal

    model.to(device)

    for epoch in range(args.epoch):
        # ========================= TRAIN =========================
        model.train()
        train_loss = 0.0
        all_targets, all_preds, all_probs = [], [], []

        for batch in train_loader:
            inputs = batch["image"].to(device)
            targets = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)              # (B, num_classes)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            probs = torch.softmax(outputs, dim=1).detach().cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(preds)
            all_probs.extend(probs)

        train_loss /= len(train_loader)
        train_bal_acc = balanced_accuracy_score(all_targets, all_preds)

        # Compute ROC AUC and Average Precision (macro)
        y_true = np.array(all_targets)
        y_prob = np.array(all_probs)[:, 1]
        try:
            roc_auc = roc_auc_score(y_true, y_prob)
            avg_prec = average_precision_score(y_true, y_prob)
        except ValueError:
            # This can happen if only one class appears in a batch/epoch
            roc_auc, avg_prec = 0.0, 0.0

        history["train_loss"].append(train_loss)
        history["train_bal_acc"].append(train_bal_acc)
        history["train_roc_auc"].append(roc_auc)
        history["train_avg_prec"].append(avg_prec)

        # ======================= VALIDATION =======================
        val_loss, val_bal_acc, val_roc_auc, val_avg_prec = validate_model(
            model, val_loader, criterion, device=device, num_classes=num_classes
        )

        history["val_loss"].append(val_loss)
        history["val_bal_acc"].append(val_bal_acc)
        history["val_roc_auc"].append(val_roc_auc)
        history["val_avg_prec"].append(val_avg_prec)

        print(
            f"[Fold {fold}] Epoch {epoch+1}/{args.epoch} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Train BalAcc: {train_bal_acc:.4f} | Val BalAcc: {val_bal_acc:.4f}"
        )

        # Save best model by validation balanced accuracy
        if val_bal_acc > best_bal_acc:
            best_bal_acc = val_bal_acc
            best_path = os.path.join(args.out_dir, f"best_model_fold_{fold}.pth")
            torch.save(model.state_dict(), best_path)
            print(f"  ✅ New best model saved for fold {fold} with Val BalAcc = {best_bal_acc:.4f}")

    # ====================== PLOT METRICS =========================
    plot_metrics(history, args.out_dir, fold)
    print(f"📊 Training curves saved in {args.out_dir} for fold {fold}")


def validate_model(model, val_loader, criterion, device="cpu", num_classes=2):
    model.eval()
    val_loss = 0.0
    all_targets, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for batch in val_loader:
            inputs = batch["image"].to(device)
            targets = batch["label"].to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()

            probs = torch.softmax(outputs, dim=1).detach().cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(preds)
            all_probs.extend(probs)

    val_loss /= len(val_loader)
    val_bal_acc = balanced_accuracy_score(all_targets, all_preds)

    y_true = np.array(all_targets)
    y_prob = np.array(all_probs)[:, 1]
    try:
        roc_auc = roc_auc_score(y_true, y_prob)
        avg_prec = average_precision_score(y_true, y_prob)
    except ValueError:
        roc_auc, avg_prec = 0.0, 0.0

    return val_loss, val_bal_acc, roc_auc, avg_prec


def plot_metrics(history, out_dir, fold):
    metrics = ["loss", "bal_acc", "roc_auc", "avg_prec"]

    for m in metrics:
        plt.figure()
        plt.plot(history[f"train_{m}"], label="Train")
        plt.plot(history[f"val_{m}"], label="Val")
        plt.xlabel("Epoch")
        plt.ylabel(m.upper())
        plt.title(f"{m.upper()} over epochs (Fold {fold})")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{m}_fold_{fold}.png"))
        plt.close()
