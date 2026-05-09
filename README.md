# ScribbleDose

This is the official repository for **ScribbleDose** ([ScribbleDose: Scribble-Guided Dose Prediction in Radiotherapy](https://arxiv.org/abs/2511.06897)). The code will be further organized and refined.

### Data Preprocessing

This project uses two types of input data: the original radiotherapy data and the corresponding scribble annotations.

The original dose prediction data, including CT images, dose distributions, and anatomical structure information, is obtained from [GDP-HMM_AAPMChallenge](https://github.com/RiqiangGao/GDP-HMM_AAPMChallenge). The scribble annotations are generated based on the original structure masks following [Scribbles4All](https://github.com/wbkit/Scribbles4All). We are preparing to release the processed scribble annotations as an open-source extension to the original dataset, and the download link will be provided once available.

Before training or testing, please specify the `npz_path` column in the corresponding metadata files:

- [`meta_data.csv`](meta_files/meta_data.csv) for training and validation.
- [`meta_data_infer_val.csv`](meta_files/meta_data_infer_val.csv) for testing.

### Environment Setup

You can set up the environment according to the configuration provided in [GDP-HMM_AAPMChallenge](https://github.com/RiqiangGao/GDP-HMM_AAPMChallenge).

### Training, Testing, and Evaluation

We provide separate scripts for model training, testing, and evaluation:

- Training: [`train_lightning.sh`](train_lightning.sh)
- Testing: [`infer_testing.sh`](infer_testing.sh)
- Evaluation: [`evaluate.py`](evaluate.py)

### Acknowledgments

We thank the authors of [GDP-HMM_AAPMChallenge](https://github.com/RiqiangGao/GDP-HMM_AAPMChallenge), [Scribbles4All](https://github.com/wbkit/Scribbles4All), and [RMT](https://github.com/qhfan/RMT) for their publicly available codebases.
