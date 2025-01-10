# Image Segmentation using U-Net
[![Python](https://img.shields.io/badge/Python-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/) 
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/) 
![Artificial Intelligence (AI)](https://img.shields.io/badge/Artificial%20Intelligence%20(AI)-orange.svg?logo=ai&logoColor=white)
![Image Segmentation](https://img.shields.io/badge/Image%20Segmentation-red.svg?logo=segmentation&logoColor=white)
[![Carvana Image Masking Challenge](https://img.shields.io/badge/Kaggle-Carvana%20Image%20Masking%20Challenge-blue.svg?logo=kaggle&logoColor=white)](https://www.kaggle.com/competitions/carvana-image-masking-challenge)
[![Cityscapes Dataset](https://img.shields.io/badge/Dataset-Cityscapes%20Dataset-00BFFF.svg?logo=data:image/png;base64,iVBORw0KGg&logoColor=white)](https://www.cityscapes-dataset.com/)
![Last Updated: December 2023](https://img.shields.io/badge/Last%20Updated-December%202023-green.svg)

## Backup of Old Project (December 2023)

This is a backup of an old project focused on training a U-Net model from scratch for semantic segmentation from scratch on the Cityscapes dataset and Carvana dataset. The images are DOWNSCALED to speed up the training process for learning purposes. The model has been trained and tested with the following results:

> [!WARNING]
> If you are looking for a high-quality model, this is NOT the place. This is only a practice exercise when I was in year 2.

> [!NOTE] 
> Training is done long ago and some parameters recorded, such as the number of epochs, may not be accurate

> [!IMPORTANT]
> This project is done few years ago from the point of writing this documentation. Back then, the training details are not documented properly and the jupyter notebook results are not saved for each experiment. The results are not excellent either. Sorry for the inconvenience.

## Setups
**Note**: 
- **Environment**: Python 3.10.8, PyTorch 2.1.2, CUDA 12.1
- **Libraries**: See `requirements-lock.txt`
- **Folder Layout**: 
  ```
  data/
    carvana/
      test/
      train/
      train_masks/
      val/
      val_masks/
    cityscapes/
      gtFine/
      leftImg8Bit/
  ```
  - Data Sources: 
    - Carvana: [![Carvana Image Masking Challenge](https://img.shields.io/badge/Kaggle-Carvana%20Image%20Masking%20Challenge-blue.svg?logo=kaggle&logoColor=white)](https://www.kaggle.com/competitions/carvana-image-masking-challenge)
      - Download from the above link
      - Unzip all zips and organize as described above
    - Cityscapes: [![Cityscapes Dataset](https://img.shields.io/badge/Dataset-Cityscapes%20Dataset-00BFFF.svg?logo=data:image/png;base64,iVBORw0KGg&logoColor=white)](https://www.cityscapes-dataset.com/)
      - Download the coarse dataset (images and the masks)
      - Unzip all zips and organize as described above
- **Models**: 
  ```bash
  models/
    sem_segmentation/
      1703922437_cityscapes/model_1703922437.h5 # cityscapes
      1703926149/model_1703926149.h5 # carvana
  ```
  Download from: [![Google Drive](https://img.shields.io/badge/Google%20Drive-Models-orange.svg?logo=googledrive&logoColor=white)](https://drive.google.com/drive/folders/1Mgb_YWV__zsQGNryXGvlOa2EaH49WERe)
- **Training scripts**: Run `main_semantic_segmentation_carvana.py` and `main_semantic_segmentation_cityscape.py` as juypter notebooks.
- **Testing scripts**: Run the `test_semantic_segmentation_carvana.py` script as jupyter notebook. Specify the paths of the h5 files in cell [8]. 

## Experiments
### Carvana Dataset (Carvana Image Masking Challenge Dataset) [![Carvana Image Masking Challenge](https://img.shields.io/badge/Kaggle-Carvana%20Image%20Masking%20Challenge-blue.svg?logo=kaggle&logoColor=white)](https://www.kaggle.com/competitions/carvana-image-masking-challenge)

Dataset information
* Original image dimension: 1918 x 1280
* Training image dimension: 160 x 240 and 320 x 480

Validation accuracies (highest)
* Validation Pixel Accuracy: **0.9955**
* Validation Dice Score: **0.9911**

<h3 align="center"> Results </h3>

Results on test set: (Top: Prediction, Bottom: Reference)

![alt text](docs/image.png)

Results on train set: (Top: Prediction, Bottom: Reference)

![alt text](docs/image2.png)

Inspection of intermediate layers: (In the order: Prediction, Output of Downsampling Block 1, Output of Bottleneck Block, Output of Upsampling Block 1, Reference)

![alt text](docs/image4.png)

<details>
<summary>Details of the experiments</summary>

```
Normalization: mean 0, std 1
Augments:
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.RandomRotation(degrees=35),

**29/12/2023 17:15: result12_**
Downscaled image dimension: 160 x 240
Batch size: 32
Learning rate: 5e-7
Decay: StepLR: step_size=5, gamma=0.85
Loss: BCEWithLogitsLoss
Epochs: 42
Final: Test accuracy: 90.9%, Avg loss: 0.403125, Test recall: 0.5831877589225769, precision: 0.980972170829773

**30/12/2023 08:40: result13_**
Downscaled image dimension: 160 x 240
Batch size: 32
Learning rate: 1e-4
Decay: ReduceLROnPlateau: patience=5
Loss: BCEWithLogitsLoss
Epochs: 50
Final: Test accuracy: 81.0%, Avg loss: 0.775568, Test recall: 0.1010913997888565, precision: 0.9727051258087158

**30/12/2023 11:33: result14_**
Downscaled image dimension: 160 x 240
Batch size: 32
Learning rate: 1e-4
Decay: ReduceLROnPlateau: patience=5
Loss: BCEWithLogitsLoss
Epochs: 100
Final:
Test accuracy: 96.6%, Avg loss: 0.210099, Test recall: 0.9129086136817932, precision: 0.927651584148407

**30/12/2023 15:47: 1703922437**
Downscaled image dimension: 160 x 240
Batch size: 16
Learning rate: 1e-4
Decay: None
Loss: BCEWithLogitsLoss
Test accuracy: 99.5%, Avg loss: 0.013175, 
Test recall: 0.9918522238731384, precision: 0.9872172474861145, dice_score: 0.9895293116569519

**30/12/2023 16:49: 1703926149**
Downscaled image dimension: 320 x 480
Batch size: 8
Learning rate: 1e-4
Decay: None
Loss: BCEWithLogitsLoss
Test accuracy: 99.6%, Avg loss: 0.009994, 
Test recall: 0.9942919611930847, precision: 0.9879629611968994, dice_score: 0.9911173582077026
```
</details>

-----

### Cityscapes Dataset (Cityscapes Dataset) [![Cityscapes Dataset](https://img.shields.io/badge/Dataset-Cityscapes%20Dataset-00BFFF.svg?logo=data:image/png;base64,iVBORw0KGg&logoColor=white)](https://www.cityscapes-dataset.com/)

* Test Pixel Accuracy: **0.8669**
* The code for evaluating the mAP for the cityscape dataset has been lost. The code in this repository does not reflect the true results.

```
1704279189_cityscapes/model_1704284109.h5: 

**Documented:** 
Test acc: 0.8402184247970581, Test loss: 0.6274673556908965
```

