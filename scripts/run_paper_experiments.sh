#!/usr/bin/env bash
# Runs every experiment of the paper in the order of the paper. Each run takes from about a minute
# (GPS models) to about 28 minutes (fusion models) on one Colab GPU. Results go to /content/results.
set -e
cd "$(dirname "$0")/../src"

# 1. vision only
python v1_resnet50.py
python v2_effnetv2.py
python v3_swin_tiny.py
python v4_convnext_base_fast.py

# 2. GPS only
python g1_linear.py
python g2_mlp.py
python g3_resmlp.py
python g4_fttransformer.py

# 3. concatenation and attention without residual connection
python fusion_v2.py C0 42
python fusion_v2.py A1_SE 42
python fusion_v2.py A4_ECA 42
python fusion_v2.py A2_UFO 42
python fusion_v2.py S1_SK_NoRes 42

# 4. attention with residual connection
python fusion_v2.py R1_SE_Res 42
python fusion_v2.py R3_ECA_Res 42
python fusion_v2.py R2_UFO_Res 42
python fusion_v2.py A3_SK 42

python collect_results_v2.py
