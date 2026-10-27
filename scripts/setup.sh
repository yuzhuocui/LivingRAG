#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -c 'import sys; assert sys.version_info[:2] == (3, 9), "Activate a Python 3.9 environment first."'
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
python -m pip install 'https://github.com/explosion/spacy-models/releases/download/en_core_web_trf-3.6.1/en_core_web_trf-3.6.1-py3-none-any.whl'
python -m pip check
python - <<'PY'
import torch
assert torch.cuda.is_available(), 'CUDA is unavailable. Check the NVIDIA driver and GPU access.'
x = torch.randn(128, 128, device='cuda')
assert torch.isfinite(x @ x).all()
torch.cuda.synchronize()
print('GPU check passed:', torch.__version__, torch.cuda.get_device_name(0))
PY
