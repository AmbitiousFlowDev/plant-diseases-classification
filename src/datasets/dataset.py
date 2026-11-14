from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class PlantDataset(Dataset):
    def __init__(self, csv_path, data_root, preprocessor):
        self.df = pd.read_csv(csv_path)
        self.root = Path(data_root)
        self.prep = preprocessor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        image_path = self.root / row["path"]

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image = self.prep.pipeline(image)

        label = int(row["label_idx"])

        return image, label