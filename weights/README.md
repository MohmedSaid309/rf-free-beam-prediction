## Model Weights

The checkpoint is available on the GitHub release **v1.0** [here](https://github.com/MohmedSaid309/rf-free-beam-prediction/releases/tag/v1.0).

| File                     | Model                                                 | Scenario 5 Top-1 / Top-3 / Top-5 | Loader                 | Size    |
| ------------------------ | ----------------------------------------------------- | -------------------------------- | ---------------------- | ------- |
| `F_A3_SK_best.pth`       | ConvNeXt-B + GPS, SK attention with residual, seed 42 | 53.70 / 88.50 / 95.47            | `--model sk_attention` | 347.2KB |

```bash
cd src
python evaluate_checkpoint.py ../weights/F_A3_SK_best.pth --model sk_attention
