#!/bin/bash
#SBATCH --job-name=garf_env_test
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=env_test_%j.out

cd /data/gpfs/projects/punim2657/GARF
source .venv/bin/activate

echo "=== SYSTEM INFO ==="
nvidia-smi
echo ""

echo "=== PYTHON & PYTORCH ==="
python -c "
import sys
print(f'Python: {sys.version}')

import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA compiled version: {torch.version.cuda}')
print(f'CUDA available: {torch.cuda.is_available()}')

if not torch.cuda.is_available():
    print('CUDA NOT AVAILABLE - stopping')
    sys.exit(1)

print(f'Device: {torch.cuda.get_device_name(0)}')
print(f'GPU count: {torch.cuda.device_count()}')
print(f'Compute capability: {torch.cuda.get_device_capability(0)}')
print(f'cuDNN: {torch.backends.cudnn.version()}')
print()

# Basic GPU compute
print('=== GPU COMPUTE TEST ===')
x = torch.randn(1000, 1000, device='cuda')
y = x @ x.T
print(f'Matrix multiply: OK ({y.shape})')

# Test bf16 (important for our fix)
x_bf16 = torch.randn(100, 100, device='cuda', dtype=torch.bfloat16)
y_bf16 = x_bf16 @ x_bf16.T
print(f'BFloat16 compute: OK')
print()

# PyTorch3D transforms
print('=== PYTORCH3D ===')
from pytorch3d import transforms
quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device='cuda')
mat = transforms.quaternion_to_matrix(quat.unsqueeze(0))
aa = transforms.matrix_to_axis_angle(mat)
print(f'quaternion_to_matrix: OK')
print(f'matrix_to_axis_angle: OK')
print()

# Flash Attention
print('=== FLASH ATTENTION ===')
import flash_attn
from flash_attn import flash_attn_func
q = torch.randn(1, 8, 64, 64, device='cuda', dtype=torch.float16)
k = torch.randn(1, 8, 64, 64, device='cuda', dtype=torch.float16)
v = torch.randn(1, 8, 64, 64, device='cuda', dtype=torch.float16)
out = flash_attn_func(q, k, v)
print(f'flash_attn_func: OK ({out.shape})')
print()

# spconv
print('=== SPCONV ===')
import spconv.pytorch as spconv
print(f'spconv: OK')
print()

# torch-scatter
print('=== TORCH-SCATTER ===')
import torch_scatter
src = torch.randn(10, 5, device='cuda')
index = torch.tensor([0, 0, 1, 1, 1, 2, 2, 3, 3, 3], device='cuda')
out = torch_scatter.scatter_mean(src, index, dim=0)
print(f'scatter_mean: OK ({out.shape})')
print()

# Full model import test
print('=== MODEL IMPORTS ===')
from assembly.models.denoiser.denoiser_flow_matching import DenoiserFlowMatching
from assembly.models.denoiser.denoiser_diffusion import DenoiserDiffusion
from assembly.models.denoiser.modules.scheduler import SE3FlowMatchEulerDiscreteScheduler
from assembly.data.breaking_bad.module import BreakingBadDataModule
print('All model imports: OK')
print()

print('=============================')
print('ALL TESTS PASSED')
print('=============================')
"
