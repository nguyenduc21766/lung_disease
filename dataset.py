from torch.utils.data import Dataset
import cv2
import numpy as np
import torch


def read_xray(path, img_size=224):
    """
    Read a chest X-ray as grayscale, resize, scale to [0,1],
    and convert to 3-channel (3, H, W) numpy array.
    """
    xray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if xray is None:
        raise FileNotFoundError(f"Could not read image at: {path}")

    # Resize to a fixed size (for ResNet)
    xray = cv2.resize(xray, (img_size, img_size))

    # Scale to [0,1]
    xray = xray.astype(np.float32) / 255.0

    # Stack to 3 channels: (3, H, W)
    xray_3ch = np.stack([xray, xray, xray], axis=0)

    return xray_3ch


class LungXrayDataset(Dataset):
    def __init__(self, dataframe, img_size=224, transform=None):
        """
        dataframe: pandas DataFrame with columns ['Name', 'Path', 'Label']
        img_size:  output image size (img_size x img_size)
        transform: optional transform applied to the image
                   (expects and returns a tensor or numpy array)
        """
        self.df = dataframe.reset_index(drop=True)
        self.img_size = img_size
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img_path = row["Path"]
        label = int(row["Label"])

        # (3, H, W) numpy float32 in [0,1]
        img = read_xray(img_path, img_size=self.img_size)

        # Apply transform or just convert to tensor
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = torch.from_numpy(img)            # -> tensor

        label = torch.tensor(label, dtype=torch.long)

        return {
            "image": img,
            "label": label
        }
