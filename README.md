# Model Comparison on Fetal Head Abnormalities Dataset
## Overview
This repository implements deep learning models (ResNet50 and MobileNetV3) for classifying fetal head ultrasound images into 14 abnormality categories. Supports:
- **Standard Augmentation**  
- **Advanced Augmentation** (CutMix augmentation)  
- **Few-Shot Learning**  
- **Zero-Shot Evaluation**
## Setup
```bash=
git clone https://github.com/sath-vik/ADL_1
cd ADL_1
bash download_dataset.sh
cd data
unzip fetal-head-abnormalities-classification.zip
cd ..
mkdir -p results
```
### Dataset structure

![image](https://hackmd.io/_uploads/SJZld6kd1l.png)

## Training
You can change the configuration of training in `train_resnet50.py` and `train_mobilenet_v3.py` and train using the following commands
```bash!
python train_resnet50.py
```
```bash!
python train_mobilenet_v3.py
```
```bash!
python zero_shot.py
```

Running each of them will generate their corresponding .json files in the `results/` folder.

## Visualization
For generating the metrics, confusion matrix and training history (loss vs epoch and validation accuracy vs epoch) run the following command
```bash!
python generate_visualizations.py
```

In the code of generate_visualizations.py select the .json file for which you want to generate visualizations.
