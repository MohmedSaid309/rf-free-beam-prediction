from common import *
set_seed(42)
model = GPSFTTransformer()
run_experiment('G4_FT_Transformer', model, mode='gps', epochs=60, batch_size=256, max_lr=1e-3,
               mixup_p=0.0, patience=10, num_workers=0)
