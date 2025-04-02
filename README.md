# Physics Informed Generative AI for High Resolution Flood Mapping
Pytorch implementation and pretrained models

## Pretrained models
This section contains the latent diffusion model checkpoints. `_gen.pth` files contain the model parameters and are needed for inference, while `_opt.pth` files contain the saved optimizer and are needed for further transfer learning. The pretrained autoencoder checkpoint can be found [here](https://drive.google.com/file/d/1MKWg5LxmM7GLK-EQI7OAfz_NyAq-nxUy/view?usp=sharing).

| Training Catchment | Download |
| :----------------: | :------: |
| Wollombi           | [checkpoint](https://drive.google.com/drive/folders/1lu39Zfs-wA01czTXvnoOOI5LSeClRZk2?usp=sharing) |
| Burnett            | [checkpoint](https://drive.google.com/drive/folders/1mp4n1uPkRVhe27OpUUnkoE726Wx2agYL?usp=sharing) |
| Chowilla           | [checkpoint](https://drive.google.com/drive/folders/1m5hnVxXkf6R8kkpJbeKljij_TqobRqSf?usp=sharing) |



## Requirements
Python 3.10.12+
```
pip install -r requirements.txt
```

## Config
Different configuration options for training and testing of models are listed in `config.json`. See `config` directory for examples of config files.

## Catchment data structure
Catchment data should be stored in this folder structure:
```
.
└── catchment_name/
    ├── train_hr/
    │   ├── Box_1_Depth_2021_03_18_00_00_00.tiff
    │   ├── Box_1_Depth_2021_03_18_00_30_00.tiff
    │   └── ...
    ├── train_lr/
    │   ├── Box_1_Depth_2021_03_18_00_00_00.tiff
    │   ├── Box_1_Depth_2021_03_18_00_30_00.tiff
    │   └── ...
    ├── test_hr/
    │   ├── Box_1_Depth_2015_04_21_00_00_00.tiff
    │   ├── Box_1_Depth_2015_04_21_00_30_00.tiff
    │   └── ...
    ├── test_lr/
    │   ├── Box_1_Depth_2015_04_21_00_00_00.tiff
    │   ├── Box_1_Depth_2015_04_21_00_30_00.tiff
    │   └── ...
    └── cropped_dems/
        ├── BOX_1_DEM.tiff
        ├── BOX_2_DEM.tiff
        └── ...
```

## Training
```
python sr.py -p train -c config.json -output-dir path/to/output/dir
```

## Testing
```
python sr.py -p test -c config.json -output-dir path/to/output/dir
```

## Acknowledgements

Code was adapted from [this implementation](https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement/tree/master) of [Image Super-Resolution via Iterative Refinement](https://arxiv.org/pdf/2104.07636)