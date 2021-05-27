#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --nodes=1 --cpus-per-task=2
#SBATCH --mem=96G
#SBATCH --partition=cpufast
#SBATCH --job-name ColBERT
#SBATCH --out=/home/ryparmar/logs/colbert-ctk-fever-v2.1-rerank.out

ml PyTorch/1.6.0-fosscuda-2019b-Python-3.7.4
source /home/ryparmar/venv/docretr/bin/activate

# Default Parameters
# lr 3*10e−6, bs 32, embedd dim 128, 200k iterations/training steps

export ROOT="/home/ryparmar/trained_models/colbert/"
export EXPERIMENT="ctk-fever-v2.1"
export INDEX_ROOT="${ROOT}indexes/"
export INDEX_NAME="${EXPERIMENT}.L2.32x200k"
export CHECKPOINT="${ROOT}${EXPERIMENT}/train.py/${EXPERIMENT}.l2/checkpoints/colbert.dnn"

export TOPK="${ROOT}${EXPERIMENT}/retrieve.py/test/unordered.tsv"

export tmp="#SBATCH --gres=gpu:1"


# export TRIPLES="/mnt/data/factcheck/CTK/par5/ctk-data/train-shuf-triples-text-rand+hard-neg+fever+synth.tsv"  #training triples
# export COLLECTION="/mnt/data/factcheck/CTK/par5/interim/collection_filtered.tsv"
# export QUERIES="/mnt/data/factcheck/CTK/par5/ctk-data/dev_queries.tsv"

export TRIPLES="${ROOT}data/train-triples-fever+rand+hard-neg+synth-shuf.tsv"  #training triples
export COLLECTION="/mnt/data/factcheck/CTK/par5/interim/collection_filtered.tsv"
export QUERIES="${ROOT}data/test_queries.tsv"

# export TRIPLES="/mnt/data/factcheck/fever/data-cs/fever-data/train.tsv"  #training triples
# export COLLECTION="/mnt/data/factcheck/fever/data-cs/fever/fever.tsv"
# export QUERIES="/mnt/data/factcheck/fever/data-cs/fever/dev_queries.tsv"

# export TRIPLES="/home/ryparmar/fever-cs-deepl/train.tsv"  #training triples
# export COLLECTION="/mnt/data/factcheck/fever/data-cs/fever/fever.tsv"
# export QUERIES="/home/ryparmar/fever-cs-deepl/dev_queries.tsv"

export MAXLEN=220
export BS=32
export DEVICES="0"
export NODES=1
export DIM=128

# Training with triples of ids -- works weirdly
# CUDA_VISIBLE_DEVICES=${DEVICES}  \
# python -m torch.distributed.launch \
#     --nproc_per_node=${NODES} -m colbert.train --amp \
#     --doc_maxlen ${MAXLEN} --mask-punctuation \
#     --bsize ${BS} --accum 1 --maxsteps 200000 \
#     --triples ${TRIPLES} \
#     --root ${ROOT} --experiment ${EXPERIMENT} \
#     --similarity l2 --run ${EXPERIMENT}.l2 \
#     --collection ${COLLECTION} \
#     --queries ${QUERIES}

# Training
# CUDA_VISIBLE_DEVICES=${DEVICES}  \
# python -m colbert.train --amp \
#     --doc_maxlen ${MAXLEN} --mask-punctuation --dim ${DIM} \
#     --bsize ${BS} --accum 1 --maxsteps 200000 \
#     --triples ${TRIPLES} \
#     --root ${ROOT} --experiment ${EXPERIMENT} \
#     --similarity l2 --run ${EXPERIMENT}.l2 

    # --checkpoint "${ROOT}fever-64dim/train.py/fever-64dim.l2/checkpoints/colbert.dnn"
    
    # ${CHECKPOINT}


### Training multiple nodes -- distributed
# CUDA_VISIBLE_DEVICES=${DEVICES}  \
# python -m torch.distributed.launch \
#     --nproc_per_node=${NODES} -m colbert.train --amp \
#     --doc_maxlen ${MAXLEN} --mask-punctuation --dim ${DIM} \
#     --bsize ${BS} --accum 1 --maxsteps 200000 \
#     --triples ${TRIPLES} \
#     --root ${ROOT} --experiment ${EXPERIMENT} \
#     --similarity l2 --run ${EXPERIMENT}.l2
    # --checkpoint "${ROOT}fever-64dim/train.py/fever-64dim.l2/checkpoints/colbert.dnn"


# Create Index
# torch.distributed.launch --nproc_per_node=${NODES} -m
# CUDA_VISIBLE_DEVICES=${DEVICES} OMP_NUM_THREADS=2 \
# python -m \
#     colbert.index --amp --doc_maxlen ${MAXLEN} --mask-punctuation --bsize ${BS} \
#     --checkpoint ${CHECKPOINT} \
#     --collection ${COLLECTION} --dim ${DIM} \
#     --index_root ${INDEX_ROOT} --index_name ${INDEX_NAME} \
#     --root ${ROOT} --experiment ${EXPERIMENT}


# # FAISS Indexing for end-to-end retrieval
# !NOTE: currently works only on the cpu (make sure partition=cpu)
# !NEED TO INVESTIGATE MORE
# estimate: min 50 hours using single gpu, single node and adding only via cpu index - comment lines 44-51 in faiss_index.py
# python -m colbert.index_faiss \
#    --index_root ${INDEX_ROOT} --index_name ${INDEX_NAME} \
#    --partitions 32768 --sample 0.15 --slices 1 \
#    --root ${ROOT} --experiment ${EXPERIMENT}


# # # Retrieval
###  faiss_depth = top-k documnents returned from index
# python -m colbert.retrieve \
#      --amp --doc_maxlen ${MAXLEN} --mask-punctuation --bsize ${BS} \
#      --queries ${QUERIES} --dim ${DIM} \
#      --nprobe 32 --partitions 32768 --faiss_depth 1000 \
#      --index_root ${INDEX_ROOT} --index_name ${INDEX_NAME} \
#      --checkpoint ${CHECKPOINT} \
#      --root ${ROOT} --experiment ${EXPERIMENT} \
#      --batch --retrieve_only

python -m colbert.rerank \
    --amp --doc_maxlen ${MAXLEN} --mask-punctuation --bsize ${BS} \
    --queries ${QUERIES} --dim ${DIM} \
    --partitions 32768 --depth 500 \
    --index_root ${INDEX_ROOT} --index_name ${INDEX_NAME} \
    --checkpoint ${CHECKPOINT} \
    --root ${ROOT} --experiment ${EXPERIMENT} \
    --batch --log-scores --topk ${TOPK}

# If you have a large set of queries (or want to reduce memory usage), use batch-mode retrieval and/or re-ranking. 
# This can be done by passing --batch --only_retrieval to colbert.retrieve and passing 
# --batch --log-scores to colbert.rerank alongside --topk with the unordered.tsv output of this retrieval run.
