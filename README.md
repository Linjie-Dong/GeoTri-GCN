# GeoTri-GCN

Code for **Geometric Augmentation and Adaptive Three-Stream Temporal Modeling for Cross-Subject Skeleton-Based Action Recognition**.

Authors: Linjie Dong and Xiuhui Wang, China Jiliang University.

## Scope of this release

This source release includes GeoTri-GCN and matched EfficientGCN-B0 configurations for NTU60 and NTU120 cross-subject evaluation. Configuration names identify the model and dataset, for example `GeoTri-GCN_NTU60` and `EfficientGCN-B0_NTU60`.

GeoTri-GCN combines sequence-consistent bone-length perturbation, tri-axis stream fusion (TASF), and selective multi-scale temporal refinement (SMTR). Shared source files also contain optional identity-adversarial, bone-length normalization and paired-consistency routines; none is enabled by the released GeoTri-GCN configurations.

The release preserves the model and training logic. Configuration names, output paths, comments and diagnostic labels use descriptive names; dataset paths are portable. Validation covers synthetic CPU smoke checks, not benchmark accuracy reproduction. Historical training environments may differ from the tested release environment.

Datasets, pretrained weights, experimental logs, manuscript files and later experimental runners are not included. This source-only release does not report benchmark scores.

## Attribution and license

This project builds on EfficientGCN by Yi-Fan Song and colleagues. The upstream MIT license and copyright notice are preserved in `LICENSE`. Please cite the underlying work:

Yi-Fan Song, Zhang Zhang, Caifeng Shan, and Liang Wang. *Constructing Stronger and Faster Baselines for Skeleton-Based Action Recognition*. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(2), 1474-1488, 2023. [Publisher page](https://ieeexplore.ieee.org/document/9729609).

TASF draws on adaptive feature reweighting, and SMTR adapts selective-kernel branch selection. See the manuscript for the underlying methods and attribution. A GeoTri-GCN publication DOI is not yet listed in this release.

## Environment

The release was technically checked with Python 3.10.20, PyTorch 2.8.0+cu128 and the dependency versions in `requirements.txt`.

```bash
conda create -n geotri-gcn python=3.10
conda activate geotri-gcn
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python scripts/smoke_release.py
```

The PyTorch command above is for a CUDA 12.8 build; choose a compatible build/driver for your hardware. Model and feeder smoke checks run on CPU. The original training/evaluation runner is intended for NVIDIA GPU use, and performs an NVML memory check; CPU training and end-to-end throughput are not validated here.

## Data

Obtain NTU RGB+D 60 and NTU RGB+D 120 from the [official ROSE Lab page](https://rose1.ntu.edu.sg/dataset/actionRecognition/) under its access conditions. Do not redistribute the raw skeleton files through this repository.

The research uses the NTU RGB+D and NTU RGB+D 120 Action Recognition Datasets provided by ROSE Lab, Nanyang Technological University, Singapore. Cite both dataset papers when using these data.

Edit `dataset_args.ntu` in each YAML configuration to set:

- `ntu60_path`: directory containing the NTU60 `.skeleton` files;
- `ntu120_path`: directory containing the additional NTU120 setup S018-S032 `.skeleton` files (not a second copy of NTU60);
- `root_folder`: output directory for generated arrays.

Use directories containing only the expected skeleton files. The NTU120 reader combines both raw directories. The reader's fixed subject lists and `src/reader/ignore.txt` define the split/filtering logic; inspect generated counts for your downloaded archive rather than assuming a nominal total.

Generate each processed dataset once:

```bash
python main.py -c EfficientGCN-B0_NTU60 -gd
python main.py -c EfficientGCN-B0_NTU120 -gd
```

Expected directory structure:

```text
data/processed/original/ntu-xsub/
    train_data.npy
    train_label.pkl
    eval_data.npy
    eval_label.pkl
data/processed/original/ntu-xsub120/
    train_data.npy
    train_label.pkl
    eval_data.npy
    eval_label.pkl
```

The feeder uses joint/velocity/bone streams, 288 input frames and two person slots. Geometry augmentation is active only during GeoTri-GCN training.

## Training and continuation

Run from the repository root. Use a different output directory for each configuration and seed:

```bash
# NTU60 paired seed 1
python main.py -c EfficientGCN-B0_NTU60 -s 1 -g 0 -w outputs/EfficientGCN-B0_NTU60_seed1
python main.py -c GeoTri-GCN_NTU60 -s 1 -g 0 -w outputs/GeoTri-GCN_NTU60_seed1

# NTU120 paired seed 1
python main.py -c EfficientGCN-B0_NTU120 -s 1 -g 0 -w outputs/EfficientGCN-B0_NTU120_seed1
python main.py -c GeoTri-GCN_NTU120 -s 1 -g 0 -w outputs/GeoTri-GCN_NTU120_seed1

# Resume after interruption, keeping config, seed and output directory unchanged
python main.py -c GeoTri-GCN_NTU60 -s 1 -g 0 -w outputs/GeoTri-GCN_NTU60_seed1 -r
```

Repeat with matching seeds and distinct directories when comparing runs. Checkpoints are saved at epoch boundaries; interruption within an epoch loses that epoch's partial work. Resume restores model, optimizer, scheduler and recorded RNG states and checks the saved experiment fingerprint. Do not change source/config/data in the middle of a run. Do not run two jobs against the same output directory.

Default training: SGD, batch size 16, 70 epochs, 10-epoch warm-up and cosine learning-rate decay. Evaluation runs every five epochs through epoch 60, then every epoch through 70. The saved best model maximizes Top-1 accuracy on the configured `eval` split. This split is the benchmark test split, **not an independent validation set**; report this selection rule, use the same rule for the baseline, and do not present the selected value as an untouched hold-out estimate.

## Evaluation

```bash
python main.py -c GeoTri-GCN_NTU60 -s 1 -g 0 -w outputs/GeoTri-GCN_NTU60_seed1 -e
```

Select the numbered saved run when prompted. Do not press Enter to evaluate random initialization. A run directory contains its best checkpoint, configuration and recognition results; the work-directory root holds the last epoch checkpoint for continuation. Only load trusted checkpoint/pickle files. The runner loads full PyTorch checkpoint objects (`weights_only=False`).

## Main implementation files

| Component | File |
|---|---|
| Training, evaluation and checkpoint control | `main.py`, `src/processor.py`, `src/initializer.py`, `src/utils.py` |
| Geometry perturbation and J/V/B construction | `src/dataset/ntu_feeder.py` |
| TASF and SMTR | `src/model/fusions.py` |
| Integration into EfficientGCN | `src/model/nets.py` |
| NTU parsing, subject splits and exclusions | `src/reader/ntu_reader.py`, `src/reader/ignore.txt` |

Identity-probe, donor-transfer and efficiency analysis scripts are outside this release. Reproducing those tables requires the corresponding scripts and checkpoint mappings in addition to this package.
