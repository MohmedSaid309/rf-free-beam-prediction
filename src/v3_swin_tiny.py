from common import *
set_seed(42)
model = VisionOnly('swin_tiny_patch4_window7_224')
run_experiment('V3_Swin_T', model, mode='vision', epochs=15, batch_size=32, max_lr=5e-4, mixup_p=0.3)
