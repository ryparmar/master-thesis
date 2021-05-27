#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name eval-ict
#SBATCH --out=/home/ryparmar/logs/eval_on_fever_dev.out

ml PyTorch/1.6.0-fosscuda-2019b-Python-3.7.4
source /home/ryparmar/venv/docretr/bin/activate
# ml faiss/1.6.1-fosscuda-2019b-Python-3.7.4

python src/main_eval.py --mode finetuning \
                        --claims_path /mnt/data/factcheck/fever/data-cs/fever-data \
                        --articles_path /mnt/data/factcheck/fever/data-cs/fever/fever.db \
                        --articles_chunks_path /home/ryparmar/trained_models/ids-chunks-288-finetuning-feverwiki.pkl \
                        --continue_training /home/ryparmar/trained_models/mbert_pre_wiki_10ep_1e-5_288 \
                        --test_batch_size 32