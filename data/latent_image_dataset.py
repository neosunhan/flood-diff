from torch.utils.data import Dataset
import os
import json
import torch

class LatentImageDataset(Dataset):
    def __init__(self, low_res_folder, high_res_folder, dem_folder, transform=None, data_len=-1, dem=True):
        self.low_res_folder = low_res_folder
        self.high_res_folder = high_res_folder
        self.dem_folder = dem_folder
        self.transform = transform
        self.filenames = [f for f in os.listdir(high_res_folder) if os.path.isfile(os.path.join(high_res_folder, f))]  
        if data_len != -1:
            data_len = min(data_len, len(self.filenames))
            self.filenames = self.filenames[:data_len]
        self.dem = dem

    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        low_res_path = os.path.join(self.low_res_folder, self.filenames[idx])
        low_res_image = torch.load(low_res_path, weights_only=False)
        high_res_path = os.path.join(self.high_res_folder, self.filenames[idx])
        high_res_image = torch.load(high_res_path, weights_only=False)

        if self.dem:
            get_dem_name = self.filenames[idx].split("_")
            dem_name = get_dem_name[0] + "_" + get_dem_name[1] + "_DEM.pt"
            dem_path = os.path.join(self.dem_folder, dem_name)
            dem_image = torch.load(dem_path, weights_only=False)
        else:
            dem_image = torch.tensor(0)  # place holder, not used

        profile = None

        if self.transform:
            low_res_image = self.transform(low_res_image)
            high_res_image = self.transform(high_res_image)

        return {
            "filename": self.filenames[idx],
            "profile": json.dumps(profile),
            "SR": low_res_image,
            "HR": high_res_image,
            "DEM": dem_image,
        }