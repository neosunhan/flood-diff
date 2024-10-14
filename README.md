# Physics Informed Generative AI for High Resolution Flood mapping

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