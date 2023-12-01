## Environment setup

```
conda env create -f environment.yml

conda activate h4g
```

## Data setup

We assume that the data is stored in a `data/` folder in the **root** directory of this repository. We assume that our labels are stored in `labels.csv`, the images in `images/`, and the weather data is stored in `era5_land_t2m_pev_tp.csv`. 

## Extracting the ts-fresh time series features

First we need to extract the time-series features with tsfresh. After we executed this, you should have two additional files in the **root** `data/` folder: `ts_features_full_narm.csv` and `relevant_features.csv`. Note that feature extraction may take a long time (~20 min per 4000 datapoints). 

```
python3 jan/ts_fresh_feature_extraction.py
TODO: Figure out how to run R scripts from the command line 
```

## Training a pure image model

Creates a train-validation-test split, trains a vision model on the images with earlystopping on the validation set. Evaluates the model on test set, prints some basic metrics and stores the outputs in the pickle file `damage_convnext_tiny.pkl` in `data/outputs/` folder in the **root** directory. The best performing model on the validation set will also be saved in in `data/lightning_logs/` folder in the **root** directory. 

```
python3 main.py --task damage --is_multi_modal False
```

Note: You can also train for the binary task `extent` by choosing with ```python3 main.py --task extent --is_multi_modal False```

Note: This current implementation supports and per default uses multi-gpu training, evaluation, and inference if the resources are available. 

## Training a multimodal time-series and image model

Creates a train-validation-test split, trains a multi-modal model on the images and time-series features with earlystopping on the validation set. Evaluates the model on test set, prints some basic metrics and stores the outputs in the pickle file `damage_mm_convnext_tiny.pkl` in `data/outputs/` folder in the **root** directory. 

```
python3 main.py --task damage --is_multi_modal False
```

Note: You can also train for the binary task `extent` by choosing with ```python3 main.py --task extent --is_multi_modal True```

Note: This current implementation supports and per default uses multi-gpu training, evaluation, and inference if the resources are available. 

## Viewing training logs

The training logs can be viewed with the ```tensorboard.ipynb``` notebook. 

Note: If you are running this on a remote machine, you need to forward both the jupyter-notebook and tensorboard ports. Here is an example ssh config and a [tutorial](https://www.digitalocean.com/community/tutorials/how-to-configure-custom-connection-options-for-your-ssh-client) on how to work with ssh configs. 

```
Host myserver
    HostName myserver.com
    # 6123 -> tensorboard, 8123 -> jupyter-notebook
    LocalForward 6123 localhost:6123
    LocalForward 8123 localhost:8123
    User username
```

You can start the jupyter-notebook in the following way on your remote server.

```
jupyter-notebook --port 8123 --no-browser
```

## Running inference

Assuming you trained a model and you saved the checkpoint in a specific path. Adding `--ckpt_path` flag will only evalute the model.

Assuming you have trained a image model for the extent task, and it is stored in `../data/checkpoints/convnext_tiny/extent/model-epoch=00-val/F1Score=0.64.ckpt`, you can evaluate your model and create predictions with the following command

```
python3 main.py --task extent --is_multi_modal False --ckpt_path ../data/checkpoints/convnext_tiny/extent/model-epoch=00-val/F1Score=0.64.ckpt
```


## Analyzing the results 

The results of the test dataset can be analyzed using the `result_analysis.ipynb` notebook. 
Given results of the model in a dictionary as generated automatically by our framework, this notebook allows for a more in-depth analysis of the model's performance.
It visualizes basic metrics such as accuracy and f1 score, the calibration plot and the confusion matrix.
Finally, it can list the best and worst predictions and visualize their corresponding images

Caution: any changes in the training process such as the split fractions or seed need to be reflected in the notebook!


## Proof of concept website for deployment

TODO: Roman
