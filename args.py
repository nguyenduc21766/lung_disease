import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Model training options')

    # Backbone selection
    parser.add_argument(
        '--backbone',
        type=str,
        default='resnet18',
        choices=['resnet18', 'resnet34', 'resnet50'],
        help='Backbone CNN architecture'
    )

    # Where the CSVs (folds + test) live
    parser.add_argument(
        '--csv_dir',
        type=str,
        default='data/CSVs',
        help='Directory containing train/val/test CSV files'
    )

    # Training hyperparameters
    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        help='Learning rate'
    )
    parser.add_argument(
        '--epoch',
        type=int,
        default=10,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        choices=[16, 32, 64],
        help='Batch size'
    )

    # Output directory for models and plots
    parser.add_argument(
        '--out_dir',
        type=str,
        default='outputs',
        help='Output directory for saved models and training curves'
    )

    # Pneumonia project: 2 classes (0 = Normal, 1 = Pneumonia)
    parser.add_argument(
        '--num_classes',
        type=int,
        default=2,
        help='Number of output classes'
    )

    return parser.parse_args()
