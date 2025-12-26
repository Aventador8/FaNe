# FaNe
[AAAI-2026] Official code for "[FaNe: Towards Fine-Grained Cross-Modal Contrast with False-Negative Reduction and Text-Conditioned Sparse Attention](https://arxiv.org/abs/2511.12215)"

![framework](docs/overview.jpg)

### Installation
To install Python dependencies:
```
pip install -r requirements.txt
```
### Data Preparation

#### Dataset Downloading
- **MIMIC-CXR**: We downloaded the [MIMIC-CXR-JPG](https://physionet.org/content/mimic-cxr-jpg/2.0.0/) dataset as the radiographs. Paired medical reports can be downloaded in [MIMIC-CXR](https://physionet.org/content/mimic-cxr/2.0.0/mimic-cxr-reports.zip).

- **CheXpert**: We downloaded the [CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/) dataset which consisting of 224,316 chest radiographs of 65,240 patients.

- **RSNA**: We used the stage 2 of RSNA dataset in [Kaggle](https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge/data). 

- **COVIDx**: We used the version 6 of COVIDx dataset in [Kaggle](https://www.kaggle.com/datasets/andyczhao/covidx-cxr2).

- **SIIM**: We downloaded the stage 1 of SIIM dataset in [Kaggle](https://www.kaggle.com/competitions/siim-acr-pneumothorax-segmentation/data).

- **Object-CXR**: We downloaded the object-CXR dataset in its [official website](https://academictorrents.com/details/fdc91f11d7010f7259a05403fc9d00079a09f5d5).

After downloading datasets, please check if the path in `fane/constants.py` is correct.
#### MIMIC-CXR Dataset

1. Download the Version 2 of the MIMIC-CXR-JPG from `https://physionet.org/content/mimic-cxr-jpg/2.0.0/` to `<image_dataset_path>`

2. Download the reports from MIMIC-CXR `https://physionet.org/content/mimic-cxr/2.0.0/` to `<report_dataset_path>`

3. Run scripts to make json file for pre-training

```bash
cd codes
python fane/data/pretrain/mimiccxr.py build --root_image_path <image_dataset_path> --root_report_path <report_dataset_path> --save_path <dataset_json_path> --meta_csv_path <meta_csv_path>
```

4. Add `<dataset_json_path>` to `configs/data/mimiccxr.yaml`

```yaml
_target_: fane.data.pretrain.mimiccxr.MimicCxrDataset
dataset_path: <dataset_json_path> # update this line
image_transform: ???
text_transform: ???
num_colors: 3
rate: 1.0
```

### Pre-training

Run

```bash
cd codes/
python codes/scripts/pre_train.py +experiments/pre_train=train_fane
```
We train our framework 50 epochs on 2 pieces of RTX 4090 GPUs with batch size of 98. It takes about *1 day* to pre-train this model. 

### Finetune on downstream tasks
We evlauate the performance of FaNe framework on three downstream tasks: classification, object detection and semantic segmentation. 

#### Linear classification
We evaluate linear classification performance of our model using this command:
```
cd codes/fane/downstream/FaNe
CUDA_VISIBLE_DEVICES=1 python fane_classification.py --gpus 1 --dataset chexpert --data_pct 0.01
```
We can use `--dataset` to set specific dataset for finetuning. Here, 3 datsets are available: chexpert, rsna and covidx.
We can use `--data_pct` to set the fraction of training data for finetuning.

To run all experiments for this detection task:
```
sh run_cls_funetune.sh
```
#### Object detection
We evaluate object detection performance of our model using this command:
```
cd codes/fane/downstream/FaNe
CUDA_VISIBLE_DEVICES=0 python fane_detector.py --devices 1 --dataset rsna --data_pct 1 --learning_rate 5e-4
```
Here, 2 datsets are available: rsna and object_cxr.
To run all experiments for this detection task:
```
sh run_det_funetune.sh
```
#### Semantic segmentation
We evaluate semantic segmentation performance of our model using this command:
```
cd codes/fane/downstream/FaNe
CUDA_VISIBLE_DEVICES=0 python fane_segmenter.py --gpus 1 --data_pct 1 --dataset rsna --batch_size 16 --learning_rate 5e-4
```
Here, 2 datsets are available: rsna and siim.

To run all experiments for this detection task:
```
sh run_seg_funetune.sh
```

### Acknowledgement
Some of the code is borrowed from [MGCA](https://github.com/HKU-MedAI/MGCA), [PRIOR](https://github.com/QtacierP/PRIOR). Thanks for their great work.

### Citation
If you find this work useful in your research, please cite:
```
@inproceedings{peng2026align,
  title={FaNe: Towards Fine-Grained Cross-Modal Contrast with False-Negative Reduction and Text-Conditioned Sparse Attention},
  author={Zhang, Peng and Lai, Zhihui and Chen, Wengting and Wu, Xu and Kong, Heng},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2026}
}
```


