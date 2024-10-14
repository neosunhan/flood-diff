# Physics Informed Generative AI for High Resolution Flood mapping

## Requirements
Python 3.10.5+
```
pip install -r requirements.txt
```

## Config
Different configuration options for training and testing of models are listed in `config.json`. See `config` directory for examples of config files.

## Training
```
python sr.py -p train -c config.json -output-dir path/to/output/dir
```



## Acknowledgements

Code was adapted from [this implementation](https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement/tree/master) of [Image Super-Resolution via Iterative Refinement](https://arxiv.org/pdf/2104.07636)