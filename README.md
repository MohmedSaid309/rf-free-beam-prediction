# RF-Free Vision–Location Beam Prediction with Feature-Wise Attention Fusion

Official code for the paper *"RF-Free Vision–Location Beam Prediction with Feature-Wise Attention Fusion for 6G mmWave Systems"* (Mohamed Said, Nour Eldin Elmadany — AASTMT, Alexandria).

A ConvNeXt-Base visual branch (5 stacked frames, 15 channels) and a GPS motion-feature branch are concatenated and re-weighted by a feature-wise attention gate to predict the optimal beam out of a 64-beam codebook. **No RF / received-power input is used.** Training uses DeepSense 6G Scenarios 1–4; testing uses the unseen Scenario 5.

## Results (Scenario 5, Top-k accuracy %)

| Exp. | `--variant` | Configuration | Top-1 | Top-3 | Top-5 |
|---|---|---|---|---|---|
| 1 | `baseline` | ConvNeXt + GPS (no attention) | 47.95 | 82.75 | 92.03 |
| 2 | `coord` | + Coordinate (1-D) | 47.87 | 80.31 | 89.55 |
| 3 | `sk` | + SK (1-D) | 48.65 | 83.80 | 91.81 |
| 4 | `eca` | EfficientNetV2-RW-M + ECA (different backbone) | 50.26 | 84.36 | 93.99 |
| 5 | `ufo` | + UFO-style gate | 50.52 | 84.67 | 94.21 |
| 6 | `se` | **+ SE-style gate (proposed)** | **51.31** | **85.50** | 93.29 |

Single run each (seed not fixed in the original runs). Differences below ~1 point are within the binomial standard error for N_test = 2,296.

## Setup

```bash
git clone https://github.com/USERNAME/REPO.git
cd REPO
pip install -r requirements.txt
```

## Data

Download DeepSense 6G Scenarios 1–5 from <https://www.deepsense6g.net/> (the dataset is **not** redistributed here) and arrange them as:

```
<data_root>/
  Scenario1/Scenario1/scenario1.csv  (+ image and GPS files referenced by the CSV)
  ...
  Scenario5/Scenario5/scenario5.csv
```

The CSV must contain the `index`, `unit1_rgb`, `unit2_loc` (or `unit1_loc`) and `beam_index` columns.

## Train and evaluate

```bash
python train.py --variant se --data_root /path/to/data --seed 42
```

`--variant` is one of `baseline | coord | sk | eca | ufo | se`. Each run trains on Scenarios 1–4 (90/10 split, seed 42), keeps the checkpoint with the lowest validation loss, evaluates on Scenario 5, and writes `results/best_<variant>.pth` and `results/submission_scenario5_<variant>_seed<seed>.csv`.

Hyper-parameters per variant (peak LR, MixUp probability, activation, dropout) are defined in `src/models.py` (`VARIANTS`) and listed in Table I of the paper.

## Pre-trained checkpoints

Download links: _add the Releases / Google Drive link here_.

## Citation

```bibtex
@inproceedings{said2026rffree,
  title     = {RF-Free Vision--Location Beam Prediction with Feature-Wise Attention Fusion for 6G mmWave Systems},
  author    = {Said, Mohamed and Elmadany, Nour Eldin},
  year      = {2026}
}
```

## License

MIT (code only). The DeepSense 6G dataset has its own license.
