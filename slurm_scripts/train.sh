#!/bin/bash
#SBATCH --job-name=voxtral_train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=slurm_out/train/%x_%j.out
#SBATCH --error=slurm_out/train/%x.%j.err

#SBATCH -A thes2173
#SBATCH --partition c23g

# Create the output directory if it doesn't exist
mkdir -p slurm_out

echo "------------------------------------- Assigned Resources -------------------------------------"
echo "GPUs requested on this node: $SLURM_GPUS_ON_NODE"
echo "Assigned GPU IDs (CUDA_VISIBLE_DEVICES): $CUDA_VISIBLE_DEVICES"
scontrol show job -d $SLURM_JOB_ID | grep CPU_IDs
echo "--------------------------------------------------------------------------"

echo "--------------------------------------Loading modules...--------------------------------------"
module load FFmpeg
module load GCCcore/13.3.0
module load CUDA/12.6.3
module load cuDNN/9.8.0.87-CUDA-12.6.3
module load GCC/13.3.0
module load Python/3.12.3
echo "--------------------------------------Finished -------------------------------------------------------------------------------"
cd ~/voxtral_finetune/voxtral_finetune

VENV_DIR="$PWD/.venv"
PYTHON="$VENV_DIR/bin/python"
ACCELERATE_EXEC="$VENV_DIR/bin/accelerate"

source .venv/bin/activate

# export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
# export CPATH=$VIRTUAL_ENV/include/python3.12:$CPATH
# export LIBRARY_PATH=$CUDA_HOME/lib64/stubs:$LIBRARY_PATH
# export VENV_INCLUDE="$VIRTUAL_ENV/include/python3.12"

echo "--------------------------------------gcc------------------------------------------------------------------------"
which gcc
gcc --version
echo "LD_LIBRARY_PATH is: $LD_LIBRARY_PATH"
echo "CPATH is $CPATH"
echo "Library path is $LIBRARY_PATH"
echo "--------------------------------------------------------------------------------------------------------------"

# rm -rf ~/.triton/cache
# echo "--------------------------------------Removed .triton cache ---------------------------------------"

if [ ! -f ~/.cache/huggingface/accelerate/default_config.yaml ]; then
    echo "Configuring accelerate single-GPU, bf16"
    "$ACCELERATE_EXEC" config default\
            --mixed_precision "bf16" \
            --num_processes 1 \
            --distributed_type "NO"
fi

echo "Starting main.py"

"$ACCELERATE_EXEC" launch ./src/voxtral_finetune/main.py \
    /home/ve001107/voxtral_finetune/voxtral_finetune/config/voxtral-3B-mini-uka_data_0122.yaml
    
    