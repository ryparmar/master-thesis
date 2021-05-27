#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1 --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --job-name two-tower-pretraining
#SBATCH --out=/home/ryparmar/logs/mbert_fin_ctk_v2.1_20ep_ict+bfs_5e-6_288.out

ml PyTorch/1.6.0-fosscuda-2019b-Python-3.7.4
# ml PyTorch/1.8.0-fosscuda-2019b-Python-3.7.4
source /home/ryparmar/venv/docretr/bin/activate
# ml faiss/1.6.1-fosscuda-2019b-Python-3.7.4

export MODE="finetuning"

python src/main.py --mode ${MODE} \
                     --task BFS,ICT \
					 --claims_path /mnt/data/factcheck/CTK/dataset/v2.1/nfc \
                     --articles_path /mnt/data/factcheck/CTK/par5/interim/ctk_filtered.db \
                     --articles_chunks_path /mnt/data/factcheck/ict_chunked_data/ids-chunks-288-${MODE}-ctk_filtered.pkl \
                     --epoch 20 \
                     --model_weight /home/ryparmar/trained_models/mbert_fin_ctk_v2.1_20ep_ict+bfs_5e-6_288 \
                     --learning_rate 5e-6 \
                     --continue_training  /home/ryparmar/trained_models/mbert_pre_ctk_10ep_ict+bfs_1e-5_288 \
                     --bs 64 \
                     --test_bs 128

					#  --claims_path /mnt/data/factcheck/CTK/par5/ctk-data \


# python src/main.py --mode ${MODE} \
#                     --claims_path /mnt/data/factcheck/CTK/par4/ctk-data \
#                     --articles_path /mnt/data/factcheck/CTK/par4/interim/ctk_filtered.db \
#                     --articles_chunks_path /mnt/data/factcheck/ict_chunked_data/ids-chunks-288-finetuning-ctk_filtered.pkl \
#                     --epoch 10 \
#                     --learning_rate 1e-6 \
#                     --model_weight /home/ryparmar/trained_models/mbert_wiki_pre_10ep-bfs_10ep-ict_1e-5_288_best \
#                     --continue_training /home/ryparmar/trained_models/mbert_wiki_pre_10ep-bfs_10ep-ict_1e-5_288 \
#                     --bs 64 \
#                     --test_bs 128

# python src/main.py --mode ${MODE} \
#                     --task ICT+BFS \
#                     --claims_path /home/ryparmar/fever-cs-deepl \
#                     --articles_path /mnt/data/factcheck/fever/data-cs/fever/fever.db \
#                     --articles_chunks_path /mnt/data/factcheck/ict_chunked_data/ids-chunks-288-${MODE}-fever.pkl \
#                     --epoch 20 \
#                     --model_weight /home/ryparmar/trained_models/mbert_fever_finetuning_20ep_bfs+ict_5e-6_288_deepl \
#                     --learning_rate 5e-6 \
#                     --continue_training /home/ryparmar/trained_models/mbert_wiki_pre_10ep-bfs_10ep-ict_1e-5_288 \
#                     --bs 64 \
#                     --test_bs 128

# PRETRAINING ON WIKIPEDIA
# --articles_path /mnt/data/factcheck/Wikipedia/wiki_cs.json \

# python src/main.py --mode 'pretraining' \
#                    --task BFS+ICT \
#                    --claims_path /mnt/data/factcheck/fever/data-cs/fever-data \
#                    --articles_path /mnt/data/factcheck/fever/data-cs/fever/fever.db \
#                    --articles_chunks_path /mnt/data/factcheck/ict_chunked_data/ids-chunks-288-finetuning-fever.pkl \
#                    --epoch 1 \
#                    --learning_rate 1e-5 \
#                    --model_weight debug.w \
#                    --bs 64 \
#                    --test_bs 128
