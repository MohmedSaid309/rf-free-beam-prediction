from common import *
set_seed(42)
model = VisionOnly('efficientnetv2_rw_m')
run_experiment('V2_EffNetV2_RW_M', model, mode='vision', epochs=15, batch_size=32, max_lr=2e-3, mixup_p=0.2)
