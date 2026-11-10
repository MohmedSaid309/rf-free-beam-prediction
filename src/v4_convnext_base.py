from common import *
set_seed(42)
model = VisionOnly('convnext_base')
run_experiment('V4_ConvNeXt_B', model, mode='vision', epochs=15, batch_size=32, max_lr=1e-3, mixup_p=0.3)
