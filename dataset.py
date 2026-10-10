import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class DeepSenseMultiModalDataset(Dataset):
    """Images (5 stacked frames) + GPS motion vector -> beam index.

    Expected layout: <root_path>/<ScenarioN>/<ScenarioN>/<scenarion>.csv
    No RF/power data is used.
    """

    def __init__(self, scenario_list, root_path, transform=None, stride=1, window_size=5):
        self.transform = transform
        self.data_samples = []

        for scenario in scenario_list:
            scenario_dir = f"{root_path}/{scenario}/{scenario}"
            csv_file = f"{scenario_dir}/{scenario.lower()}.csv"
            if not os.path.exists(csv_file):
                continue

            df = pd.read_csv(csv_file).sort_values("index").reset_index(drop=True)

            for i in range(0, len(df) - window_size + 1, stride):
                window = df.iloc[i:i + window_size]
                img_paths = [f"{scenario_dir}/{p.strip('./')}" for p in window["unit1_rgb"].values]

                gps_col = "unit2_loc" if "unit2_loc" in window.columns else "unit1_loc"
                gps_paths = [f"{scenario_dir}/{window.iloc[j][gps_col].strip('./')}" for j in range(2)]

                try:
                    coords = []
                    for p in gps_paths:
                        with open(p, "r") as f:
                            lines = f.readlines()
                        coords.append((float(lines[0].strip()), float(lines[1].strip())))
                    d_lat = coords[1][0] - coords[0][0]
                    d_lon = coords[1][1] - coords[0][1]
                    gps_vector = [coords[0][0], coords[0][1], coords[1][0], coords[1][1],
                                  d_lat, d_lon, np.sqrt(d_lat ** 2 + d_lon ** 2), np.arctan2(d_lon, d_lat)]
                except Exception:
                    continue

                last_row = window.iloc[-1]
                beam_col = [c for c in df.columns if "beam_index" in c.lower()]
                label = int(last_row[beam_col[0]]) - 1 if beam_col else 0

                if 0 <= label < 64:
                    self.data_samples.append((img_paths, gps_vector, label))

    def __len__(self):
        return len(self.data_samples)

    def __getitem__(self, idx):
        img_paths, gps_vector, label = self.data_samples[idx]
        imgs = []
        for p in img_paths:
            img = Image.open(p).convert("RGB")
            if self.transform is not None:
                img = self.transform(img)
            if not isinstance(img, torch.Tensor):
                img = transforms.ToTensor()(img)
            imgs.append(img)
        x_images = torch.cat(imgs, dim=0)  # 5 frames x 3 channels = 15
        return (x_images,
                torch.tensor(gps_vector, dtype=torch.float32),
                torch.tensor(label, dtype=torch.long))
