from common import *
set_seed(42)
model = GPSResMLP()
run_experiment('G3_ResMLP', model, mode='gps', epochs=60, batch_size=256, max_lr=1e-3,
               mixup_p=0.0, patience=10, num_workers=0)
