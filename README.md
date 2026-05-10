# ScribbleDose

This is the official repository for **ScribbleDose** ([ScribbleDose: Scribble-Guided Dose Prediction in Radiotherapy](https://arxiv.org/abs/2511.06897)).

### Data Preprocessing

This project uses two types of input data: the original GDP-HMM AAPM radiotherapy data and the corresponding ScribbleDose annotations.

The original dose prediction data, including CT images, dose distributions, anatomical structure information, and beam-related information, should be obtained from [GDP-HMM_AAPMChallenge](https://github.com/RiqiangGao/GDP-HMM_AAPMChallenge).

The ScribbleDose annotations are generated based on the original structure masks following [Scribbles4All](https://github.com/wbkit/Scribbles4All). To avoid redistributing the original radiotherapy data, we only release the processed scribble annotations and auxiliary supervoxel maps as an open-source extension to the original dataset. The annotation files are available at [Zenodo](https://zenodo.org/records/20110954).

The released annotation files keep the same filenames and directory structure as the original `.npz` files for data alignment. A recommended data structure is shown below:

```text
/path/to/data/
├── AAPM_annotation/              # Original GDP-HMM AAPM .npz files
│   ├── HaN/
│   │   ├── train/
│   │   └── valid/
│   └── Lung/
│       ├── train/
│       └── valid/
├── ScribbleDose_annotation/      # ScribbleDose annotations downloaded from Zenodo
│   ├── HaN/
│   │   ├── train/
│   │   └── valid/
│   └── Lung/
│       ├── train/
│       └── valid/
└── AAPM_annotation_merged/       # Merged .npz files for training and testing
```

After downloading both datasets, please merge the ScribbleDose annotations into the original GDP-HMM AAPM `.npz` files using:

```bash
python utils/merge_scribble_supervoxel.py \
  --clean_root /path/to/data/AAPM_annotation \
  --scribble_root /path/to/data/ScribbleDose_annotation \
  --output_root /path/to/data/AAPM_annotation_merged \
  --apply \
  --test
```

The merged `.npz` files can then be directly used by the dataloader. The `--test` option checks whether the merged files contain both the original keys and the added `*_scribble`/`supervoxels` keys.

Before training or testing, please specify the `npz_path` column in the corresponding metadata files. Each `npz_path` should point to the merged `.npz` file under `AAPM_annotation_merged`, for example:

```text
/path/to/data/AAPM_annotation_merged/HaN/train/xxxx.npz
/path/to/data/AAPM_annotation_merged/Lung/valid/xxxx.npz
```

- [`meta_data.csv`](meta_files/meta_data.csv) for training and validation.
- [`meta_data_infer_val.csv`](meta_files/meta_data_infer_val.csv) for testing.

### Environment Setup

You can set up the environment according to the configuration provided in [GDP-HMM_AAPMChallenge](https://github.com/RiqiangGao/GDP-HMM_AAPMChallenge).

### Training, Inference, and Evaluation

After preparing the merged `.npz` files and updating the metadata files, use the following commands for training, inference, and evaluation:

```bash
# Training
bash train_lightning.sh

# Inference
bash infer_testing.sh

# Evaluation
python evaluate.py
```
### License

The source code in this repository is released under the MIT License.

The processed ScribbleDose annotations released on Zenodo are distributed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0). Users should also follow the data usage terms of the original GDP-HMM AAPM dataset.

### Acknowledgments

We thank the authors of [GDP-HMM_AAPMChallenge](https://github.com/RiqiangGao/GDP-HMM_AAPMChallenge), [Scribbles4All](https://github.com/wbkit/Scribbles4All), and [RMT](https://github.com/qhfan/RMT) for their publicly available codebases.
