from torch.utils.data import Dataset
import os
import rasterio
import torch
import numpy as np
import json

class FloodDepthDatasetWithDEM(Dataset):
    def __init__(self, low_res_folder, high_res_folder, dem_folder, max_value, max_value_dem, min_value_dem, transform=None, data_len=-1, norm_range=(-1, 1)):
        self.low_res_folder = low_res_folder
        self.high_res_folder = high_res_folder
        self.dem_folder = dem_folder
        self.max_value = max_value
        self.max_value_dem = max_value_dem
        self.min_value_dem = min_value_dem
        self.transform = transform
        self.norm_range = norm_range
        self.filenames = [f for f in os.listdir(high_res_folder) if os.path.isfile(os.path.join(high_res_folder, f))]  
        if data_len != -1:
            self.filenames = self.filenames[:data_len]

    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        high_res_path = os.path.join(self.high_res_folder, self.filenames[idx])
        low_res_path = os.path.join(self.low_res_folder, self.filenames[idx])
        get_dem_name = self.filenames[idx].split("_")
        dem_name = get_dem_name[0] + "_" + get_dem_name[1] + "_DEM.tiff"
        dem_path = os.path.join(self.dem_folder, dem_name)
        with rasterio.open(low_res_path) as src:
            low_res_image = src.read(1).astype(np.float32)
            low_res_image = (self.norm_range[1] - self.norm_range[0]) * (low_res_image / self.max_value) + self.norm_range[0]
        with rasterio.open(high_res_path) as src:
            profile = dict(src.profile)
            profile["crs"] = profile["crs"].to_string()
            profile["transform"] = profile["transform"].to_gdal()
            high_res_image = src.read(1).astype(np.float32)
            high_res_image = (self.norm_range[1] - self.norm_range[0]) * (high_res_image / self.max_value) + self.norm_range[0]
        with rasterio.open(dem_path) as src:
            dem_image = src.read(1).astype(np.float32)
            dem_image = (self.norm_range[1] - self.norm_range[0]) * (dem_image - self.min_value_dem) / (self.max_value_dem - self.min_value_dem) + self.norm_range[0]
        low_res_image = torch.from_numpy(low_res_image).unsqueeze(0)
        high_res_image = torch.from_numpy(high_res_image).unsqueeze(0)
        dem_tensor = torch.from_numpy(dem_image).unsqueeze(0)
        if self.transform:
            low_res_image = self.transform(low_res_image)
            high_res_image = self.transform(high_res_image)
        # return low_res_image, high_res_image, dem_tensor, self.filenames[idx]
        return {
            "filename": self.filenames[idx],
            "profile": json.dumps(profile),
            "SR": low_res_image,
            "HR": high_res_image,
            "DEM": dem_tensor,
        }