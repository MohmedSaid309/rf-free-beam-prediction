from common import *
set_seed(42)
model = GPSMLP()
run_experiment('G2_MLP', model, mode='gps', epochs=60, batch_size=256, max_lr=3e-3,
               mixup_p=0.0, patience=10, num_workers=0)
