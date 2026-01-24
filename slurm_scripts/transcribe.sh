#!/bin/bash
#SBATCH --job-name=voxtral_transcribe
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --output=slurm_out/transcribe/%x.%j.out
#SBATCH --error=slurm_out/transcribe/%x.%j.err

#SBATCH -A thes2173
#SBATCH --partition c23g

# Create the output directory if it doesn't exist
mkdir -p slurm_out

echo "------------------------------------- Assigned Resources -------------------------------------"
echo "GPUs requested on this node: $SLURM_GPUS_ON_NODE"
echo "Assigned GPU IDs (CUDA_VISIBLE_DEVICES): $CUDA_VISIBLE_DEVICES"
scontrol show job -d $SLURM_JOB_ID | grep CPU_IDs
echo "--------------------------------------------------------------------------"

module load FFmpeg
module load GCCcore/13.3.0
module load CUDA/12.6.3
module load cuDNN/9.8.0.87-CUDA-12.6.3
module load GCC/13.3.0
module load Python/3.12.3
echo "--------------------------------------------------------------------------"

cd ~/voxtral_finetune/voxtral_finetune

VENV_DIR="$PWD/.venv"
PYTHON="$VENV_DIR/bin/python"
ACCELERATE_EXEC="$VENV_DIR/bin/accelerate"
source .venv/bin/activate

if [ ! -f ~/.cache/huggingface/accelerate/default_config.yaml ]; then
    echo "Configuring accelerate single-GPU, bf16"
    "$ACCELERATE_EXEC" config default\
            --mixed_precision "bf16" \
            --num_processes 1 \
            --distributed_type "NO"
fi
# Use uv to run the script
# '--frozen' ensures the environment isn't modified during the run
# '--' separates uv flags from the actual command
echo "Starting transcribe.py"

"$ACCELERATE_EXEC" launch ./src/voxtral_finetune/transcribe.py \
    /home/ve001107/voxtral_finetune/voxtral_finetune/config/voxtral-3B-v0.yaml \
    --split all
    
    