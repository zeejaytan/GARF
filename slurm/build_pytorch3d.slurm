#!/bin/bash
#SBATCH --job-name=build_pytorch3d
#SBATCH --partition=sapphire
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=build_pytorch3d_%j.out

cd /data/gpfs/projects/punim2657/GARF
source .venv/bin/activate

export CUDA_HOME=/apps/easybuild-2022/easybuild/software/Core/CUDA/12.5.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Verify tools
echo "=== BUILD ENVIRONMENT ==="
nvcc --version
python --version
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')"
echo ""

# Build pytorch3d from source
echo "=== BUILDING PYTORCH3D FROM SOURCE ==="
# Set compute capabilities for the GPUs on the cluster (A100=8.0, H100=9.0, L40S=8.9)
export TORCH_CUDA_ARCH_LIST="8.0;8.9;9.0"
export FORCE_CUDA=1
export MAX_JOBS=8

# Clone into persistent location
BUILD_DIR=/data/gpfs/projects/punim2657/GARF/.pytorch3d_build
rm -rf $BUILD_DIR
git clone --depth 1 --branch v0.7.7 https://github.com/facebookresearch/pytorch3d.git $BUILD_DIR
cd $BUILD_DIR

# Build and install (non-editable)
python setup.py install 2>&1

cd /data/gpfs/projects/punim2657/GARF

echo ""
echo "=== VERIFICATION ==="
python -c "
import pytorch3d
print(f'pytorch3d version: {pytorch3d.__version__}')
from pytorch3d import transforms
print('pytorch3d.transforms: OK')
from pytorch3d.loss.chamfer import chamfer_distance
print('pytorch3d.loss.chamfer: OK')
from pytorch3d.ops import knn_points
print('pytorch3d.ops.knn_points: OK')
print('ALL PYTORCH3D IMPORTS OK')
"
