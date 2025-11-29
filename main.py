from args import get_args
import pandas as pd
import os
from dataset import LungXrayDataset
from models import MyModel
from trainer import train_model

import torch
from torch.utils.data import DataLoader


def main():
    args = get_args()
    print("Args:", args)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 5-fold training loop
    for fold in range(1, 6):
        print("\n" + "=" * 60)
        print(f"Training Fold {fold}")
        print("=" * 60)

        # CSV paths
        train_csv = os.path.join(args.csv_dir, f"fold_{fold}_train.csv")
        val_csv   = os.path.join(args.csv_dir, f"fold_{fold}_val.csv")

        # Load CSVs
        train_df = pd.read_csv(train_csv)
        val_df   = pd.read_csv(val_csv)

        # Datasets
        train_dataset = LungXrayDataset(train_df)
        val_dataset   = LungXrayDataset(val_df)

        # DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        # Create model
        model = MyModel(
            backbone=args.backbone,
            num_classes=args.num_classes
        )
        model.to(device)

        # Train this fold
        train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            fold=fold,
            device=device
        )

    print("\nTraining completed for all 5 folds.")


if __name__ == "__main__":
    main()
