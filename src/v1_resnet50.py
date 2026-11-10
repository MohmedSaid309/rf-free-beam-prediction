from common import *
set_seed(42)
model = VisionOnly('resnet50')
run_experiment('V1_ResNet50', model, mode='vision', epochs=15, batch_size=32, max_lr=1e-3, mixup_p=0.3)
