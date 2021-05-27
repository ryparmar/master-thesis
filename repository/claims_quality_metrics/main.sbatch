#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --nodes=1 --ntasks-per-node=1
#SBATCH --mem=32G
#SBATCH --partition=cpufast
#SBATCH --job-name claims_quality_metrics
#SBATCH --out=/home/ryparmar/logs/claims_quality_metrics.out

# Load modules
module load PyTorch/1.5.1-fosscuda-2019b-Python-3.7.4
source /home/ryparmar/venv/docretr/bin/activate

export DATA="/mnt/data/factcheck/CTK/dataset/v2.1/nfc"
# export DATA="/mnt/data/factcheck/CTK/par5/ctk-data"
# export DATA="/mnt/data/factcheck/fever/data-cs/fever-data"

export EXPORT="/home/ryparmar/claim-quality/ctk-v2.1"


python3 src/main.py \
        --data ${DATA} \
        --cues unigram \
        --cv \
        --export ${EXPORT}

python3 src/main.py \
        --data ${DATA} \
        --cues bigram \
        --cv \
        --export ${EXPORT}

python3 src/main.py \
        --data ${DATA} \
        --cues wordpiece \
        --cv \
        --export ${EXPORT}