"""
This project is an adoption of the following repository:
https://github.com/donggyukimc/Inverse-cloze-task

This script trains a specified transformer model on wikipedia json file using the Inverse Cloze Task (ICT).
ICT takes a chunk of continuous text and randomly selects a sentence from it. Then a batch of sentences and
contexts are each encoded with the transformer model. The model tries to predict a sentence's context.

Our objective is to minimize the cosine distance between a sentence and its context.

Supports 3 datasets:
- FEVER wiki = the first paragraph (abstract) of the wiki article
- WIKI = articles of wikipedia in full length
- CTK = CTK articles on the level of paragraphs
"""

import argparse
from tqdm import tqdm
from datetime import datetime
import logging
import sys, os

import torch 
from torch.utils.data import DataLoader, TensorDataset
import transformers

import util, eval, io_util
from model import Encoder as Model


def main(config):

    # CONFIGURATION
    if config.mode not in ['finetuning', 'pretraining']:
        config.print_help()
        sys.exit(1)

    config_dict = vars(config)

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    tokenizer = transformers.BertTokenizerFast.from_pretrained(config.bert_model)

    # changes of config_dict will change the config itself as well
    config_dict['logger'] = logger
    config_dict['date'] = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    config_dict['cls_token_id'] = tokenizer.encode(tokenizer.cls_token, add_special_tokens=False)[0]
    config_dict['pad_token_id'] = tokenizer.encode(tokenizer.pad_token, add_special_tokens=False)[0]

    logger.info("GIVEN ARGUMENTS")
    for k, v in config_dict.items():
        logger.info(f"{k}: {v}")

    model, optimizer, _, _ = instantiate_model(config, tokenizer)
    
    loss_fn = torch.nn.CrossEntropyLoss()
    metrics = eval.Metrics()
    logger.info(f'Loading and chunking articles from {config.articles_path} database...')

    dev_chunks, dev_articles_ids = util.make_chunks(config.articles_path, tokenizer, config, save_chunks=False)
    dev_chunks, dev_chunks_mask = util.process_chunks(dev_chunks, config)

    logger.info('Loading train/dev claims...')
    claims_dev, evidence_dev, labels_dev = util.load_claims('dev', config)

    logger.info('Tokenizing, padding and masking the claims...')
    claims_dev, claims_dev_mask = util.process_claims(claims_dev, tokenizer, config, _pad_max=True)
    logger.info(f"{len(claims_dev)} dev claims prepared for finetuning.")


    ## Load embedded documents
    doc_emb_path = f"/home/ryparmar/trained_models/doc-emb-{config.continue_training.split('/')[-1]}.npy"
    if os.path.exists(doc_emb_path):
        model.eval()
        eval_claim_embeddings = eval.encode_chunks(claims_dev, claims_dev_mask, model, batch_size=config.test_batch_size)
        eval_document_embeddings = io_util.load_np_embeddings(doc_emb_path)
    else:
        eval_claim_embeddings, \
        eval_document_embeddings = eval.evaluation_preprocessing(claims_dev, claims_dev_mask, dev_chunks, dev_chunks_mask, model, config)
        ### CTK PRETRAIN -- Save embedded documents
        io_util.save_np_embeddings(eval_document_embeddings, doc_emb_path)
        logger.info(f"Embeddings saved")
        model.to('cpu')

    # Evaluation
    for k in [10, 20]:
        precision, recall, f1 = eval.retriever_score(eval_claim_embeddings, eval_document_embeddings, 
                                                    evidence_dev, labels_dev, dev_articles_ids, config, k=k)
        # config.logger.info
        print(f"F1: {f1}\tRecall@{k}: {recall}\tPrecision@10: {precision}")


def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


def instantiate_model(config, tokenizer):
    configure_devices(config)
    model = Model(config)
    optimizer = transformers.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0)
    last_epoch = 0
    epoch_avg_loss = 0
    if config.continue_training:
        state_dict = torch.load(config.continue_training, map_location='cpu')
        model.load_state_dict(state_dict['model'])
        if 'optimizer_state_dict' in state_dict:
            optimizer.load_state_dict(state_dict['optimizer_state_dict'])
        last_epoch = state_dict['epoch']
        # epoch_avg_loss = state_dict['loss']
        # del state_dict # TODO TEST
    if config.use_cuda:
        model = model.cuda()
        optimizer_to(optimizer, config.device)
        model = torch.nn.DataParallel(model, device_ids=config.devices)
    return model, optimizer, last_epoch, epoch_avg_loss


def configure_devices(config):
    config.devices = [int(device) for device in range(torch.cuda.device_count())]
    config.device = config.devices[0] if config.use_cuda else "cpu"


def get_loader(data, batch_size):
    data = TensorDataset(data)
    return DataLoader(data,
                      batch_size=batch_size,
                      shuffle=True,
                      sampler=None, drop_last=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True,
                        type=str, help="mode of work: pretraining / finetuning" + 
                                        "pretraining = pretraining on the Wikipedia data" + 
                                        "finetuning = finetuning on the Fever data")
    parser.add_argument("--claims_path", 
                        default="/mnt/data/factcheck/fever/data-cs/fever-data",
                        type=str, help="Path to the claims.")
    parser.add_argument("--articles_path", 
                        default="/mnt/data/factcheck/fever/data-cs/fever/fever.db",
                        type=str, help="Path to the databse with articles.")
    parser.add_argument("--articles_chunks_path", required=False, 
                        type=str, help="Path to the chunked articles data. It will speed up the things.")
    parser.add_argument("--model_weight",
                        default="/home/ryparmar/trained_models/debug.w",
                        type=str, help="Pretrained/finetuned model will be saved here. (filename of model weights)")
    parser.add_argument("--bert_model", 
                        default="bert-base-multilingual-cased", 
                        type=str, help="Pretrained Transformer.")
    parser.add_argument("--continue_training", required=False,
                        type=str, help="Specify the path to the model which should be finetuned further.")

    ### TRAINING PARAMETERS
    parser.add_argument("--learning_rate", default=1e-5,
                        type=float)  # 1e-4 used in Weakly supervised paper for pretraining, 1e-5 for finetuning
    parser.add_argument("--max_seq", default=288, type=int)
    parser.add_argument("--epoch", default=5, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--test_batch_size", default=64, 
                        type=int, help="Can use a higher number than for batch_size since" +
                                        "model doesn't need to track gradients during evaluation")
    parser.add_argument("--remove_percent", default=0.9, type=float)
    parser.add_argument("--use_cuda", type=bool, default=True)
    main(parser.parse_args())