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
import sys
import random

import torch 
from torch.utils.data import DataLoader, TensorDataset
import transformers

import util, io_util, eval
from model import Encoder as Model

import deepspeed
import json
import wandb


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
    metrics = None

    if config.continue_training:
        state_dict = torch.load(config.continue_training, map_location='cpu')
        model.load_state_dict(state_dict['model'])
        if 'optimizer_state_dict' in state_dict:
            optimizer.load_state_dict(state_dict['optimizer_state_dict'])
            for g in optimizer.param_groups:
                g['lr'] = config.learning_rate
        
        try:
            print(f"Loaded model:\nEpochs: {state_dict['epoch']}\nLoss: {state_dict['loss']}\n", 
                  f"Recall: {state_dict['rec']}\nMRR: {state_dict['mrr']}")
        except:
            pass
        
    if config.use_cuda:
        model = model.cuda()
        optimizer_to(optimizer, config.device)
        model = torch.nn.DataParallel(model, device_ids=config.devices)
    return model, optimizer, metrics


# def prepare_model(args):
#     configure_devices(args)
#     model = Model(args)
#     metrics = None

#     optimizer_grouped_parameters = prepare_optimizer_parameters(args, model)

#     # DeepSpeed initializer handles FP16, distributed, optimizer automatically.
#     model, optimizer, _, _ = deepspeed.initialize(
#         args=args,
#         model=model,
#         model_parameters=optimizer_grouped_parameters)

#     # Overwrite application configs with DeepSpeed config
#     args.train_micro_batch_size_per_gpu = model.train_micro_batch_size_per_gpu(
#     )
#     args.gradient_accumulation_steps = model.gradient_accumulation_steps(
#     )

#     # Set DeepSpeed info
#     args.local_rank = model.local_rank
#     args.device = model.device
#     model.set_device(args.device)
#     args.fp16 = model.fp16_enabled()
#     args.use_lamb = model.optimizer_name(
#     ) == deepspeed.runtime.config.LAMB_OPTIMIZER

#     # Prepare Summary Writer and saved_models path
#     # if dist.get_rank() == 0:
#     #     summary_writer = get_sample_writer(name=args.job_name,
#     #                                        base=args.output_dir)
#     #     args.summary_writer = summary_writer
#     #     os.makedirs(args.saved_model_path, exist_ok=True)

#     return model, optimizer, metrics


# def prepare_optimizer_parameters(args, model):
#     config = args
#     deepspeed_config = json.load(
#         open(args.deepspeed_config, 'r', encoding='utf-8'))

#     param_optimizer = list(model.named_parameters())
#     param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
#     no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
#     if args.deepspeed_transformer_kernel:
#         no_decay = no_decay + [
#             'attn_nw', 'attn_nb', 'norm_w', 'norm_b', 'attn_qkvb', 'attn_ob',
#             'inter_b', 'output_b'
#         ]
#     if "weight_decay" in config["training"].keys():
#         weight_decay = config["training"]["weight_decay"]
#     else:
#         weight_decay = 0.01

#     if deepspeed_config["optimizer"]["type"] not in ["OneBitAdam"]:
#         optimizer_grouped_parameters = [{
#             'params': [
#                 p for n, p in param_optimizer
#                 if not any(nd in n for nd in no_decay)
#             ],
#             'weight_decay':
#             weight_decay
#         }, {
#             'params':
#             [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
#             'weight_decay':
#             0.0
#         }]
#     else:
#         # Because 1-bit compression cannot represent exact zero, it is required to
#         # provide a momentum mask for those params that have constant exact zeros in their
#         # momentums, otherwise the compression error would keep accumulating.
#         # For example, for bert pre-training seq 128, bert.embeddings.position_embeddings.weight
#         # always have exact zeros in its momentum for row 129 to 512, because it only
#         # learns up to seq length 128 while the model supports up to 512 seq length.
#         need_mask = ['position_embeddings.weight']
#         need_mask_p = []
#         need_mask_decay = []
#         masks = []
#         for n, p in param_optimizer:
#             if any(nd in n for nd in need_mask):
#                 mask = torch.zeros_like(p.data)
#                 for position in range(args.max_seq_length):
#                     for col in range(p.size()[1]):
#                         mask[position][col] += 1
#                 if deepspeed_config["optimizer"]["type"] == "OneBitAdam":
#                     mask = torch.flatten(mask)
#                 masks.append(mask)
#                 need_mask_p.append(p)
#                 if any(nd in n for nd in no_decay):
#                     need_mask_decay.append(0.0)
#                 else:
#                     need_mask_decay.append(weight_decay)

#         optimizer_grouped_parameters = [{
#             'params': [
#                 p for n, p in param_optimizer
#                 if not any(nd in n for nd in no_decay + need_mask)
#             ],
#             'weight_decay':
#             weight_decay
#         }, {
#             'params': [
#                 p for n, p in param_optimizer
#                 if (any(nd in n
#                         for nd in no_decay) and not any(nd in n
#                                                         for nd in need_mask))
#             ],
#             'weight_decay':
#             0.0
#         }]

#         for i_mask in range(len(need_mask_p)):
#             optimizer_grouped_parameters.append({
#                 'params': [need_mask_p[i_mask]],
#                 'weight_decay':
#                 need_mask_decay[i_mask],
#                 'exp_avg_mask':
#                 masks[i_mask]
#             })

#     return optimizer_grouped_parameters



def configure_devices(config):
    config.devices = [int(device) for device in range(torch.cuda.device_count())]
    config.device = config.devices[0] if config.use_cuda else "cpu"


def get_loader(data, batch_size):
    data = TensorDataset(data)
    return DataLoader(data,
                      batch_size=batch_size,
                      shuffle=True,
                      sampler=None, drop_last=True)


def ids2docs(ids, id2doc: dict):
    return [id2doc[int(i)] for i in ids]


def main(config):
    config_dict = vars(config)

    wandb.login()
    wandb.init(project=f'{config.mode}_{config.task}')
    wb_config = wandb.config
    wb_config.learning_rate = config.learning_rate

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    tokenizer = transformers.BertTokenizerFast.from_pretrained(config.bert_model)

    # changes of config_dict will change the config itself as well
    config_dict['logger'] = logger
    config_dict['date'] = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    config_dict['cls_token_id'] = tokenizer.encode(tokenizer.cls_token, add_special_tokens=False)[0]
    config_dict['pad_token_id'] = tokenizer.encode(tokenizer.pad_token, add_special_tokens=False)[0]
    if not config.task:
        config_dict['task'] = 'ICT+BFS'

    logger.info("GIVEN ARGUMENTS")
    for k, v in config_dict.items():
        logger.info(f"{k}: {v}")

    model, optimizer, metrics = instantiate_model(config, tokenizer)  #prepare_model(config)
    loss_fn = torch.nn.CrossEntropyLoss()

    if metrics:
        logging.info(f"Metrics initialized with pretrained model metrics:\n{metrics}")
    metrics = eval.Metrics(metrics)

    logger.info(f'Loading and chunking articles from {config.articles_path} database...')
    if config.mode == 'finetuning':  ### FINETUNING
        doc_chunks = util.make_chunks(config.articles_path, tokenizer, config, save_chunks=True)
        articles_ids = util.get_par_ids(doc_chunks) if 'CTK' in config.articles_path else list(doc_chunks.keys())

        doc_chunks, chunks_mask = util.process_chunks(doc_chunks, config)
        dev_chunks, dev_chunks_mask, dev_articles_ids = doc_chunks, chunks_mask, articles_ids

        logger.info('Loading train/dev claims...')
        claims_dev, evidence_dev, labels_dev = util.load_claims('dev', config)
        claims_train, evidence_train, labels_train = util.load_claims('train', config)
        logger.info('Removing unverifiable claims...')
        claims_train, evidence_train, labels_train = util.remove_unverifiable_claims(claims_train,
                                                                                    evidence_train,
                                                                                    labels_train, config)
        claims_dev, evidence_dev, labels_dev = util.remove_unverifiable_claims(claims_dev,
                                                                        evidence_dev,
                                                                        labels_dev, config)
        
        logger.info('Removing claims for which the evidence containing document could not be found...')        
        claims_train, \
        evidence_train, \
        labels_train = util.remove_invalid_claims(claims_train, evidence_train, labels_train, articles_ids, config)

        claims_dev, \
        evidence_dev, \
        labels_dev = util.remove_invalid_claims(claims_dev, evidence_dev, labels_dev, articles_ids, config)

        logger.info('Tokenizing, padding and masking the claims...')
        claims_train, claims_train_mask = util.process_claims(claims_train, tokenizer, config, _pad_max=True)
        claims_dev, claims_dev_mask = util.process_claims(claims_dev, tokenizer, config, _pad_max=True)
        logger.info(f"{len(claims_train)} training claims and {len(claims_dev)} dev claims prepared for finetuning.")
        # if 'CTK' in config.articles_path:
    else:  ### PRETRAINING
        # Pro pretraining neni potreba nahravat claimy - claimy se extrahuji z chunku.
        doc_chunks = util.make_chunks(config.articles_path, tokenizer, config, save_chunks=True)
        # if config.task.upper() == 'ICT':
        #     doc_chunks = [chunk for doc, chunks in doc_chunks.items() for chunk in chunks] 

        dev_chunks = util.make_chunks("/mnt/data/factcheck/fever/data-cs/fever/fever.db", 
                                        tokenizer, config, as_eval=True, save_chunks=True)
        dev_articles_ids = list(dev_chunks.keys())
        dev_chunks, dev_chunks_mask = util.process_chunks(dev_chunks, config)

        logger.info('Loading dev claims...')
        claims_dev, evidence_dev, labels_dev = util.load_claims('dev', config,
                                                     path='/mnt/data/factcheck/fever/data-cs/fever-data/dev.jsonl')
        logger.info('Tokenizing, padding and masking the claims...')
        
        claims_dev, evidence_dev, labels_dev = util.remove_unverifiable_claims(claims_dev,
                                                                        evidence_dev,
                                                                        labels_dev, config)
        claims_dev, \
        evidence_dev, \
        labels_dev = util.remove_invalid_claims(claims_dev, evidence_dev, labels_dev, dev_articles_ids, config)
        claims_dev, claims_dev_mask = util.process_claims(claims_dev, tokenizer, config, _pad_max=True)

    id2doc = {i: doc_id for i, (doc_id, _) in enumerate(doc_chunks.items())} if isinstance(doc_chunks, dict) else []
    
    loader = (get_loader(torch.tensor([i for i in range(len(claims_train))]), config.bs) if config.mode == 'finetuning'
              else get_loader(torch.tensor([i for i in range(len(doc_chunks))]), config.bs))


    # Evaluation
    logger.info("Initial evaluation check...")
    def get_sample_keys(keys: list, sample=0.3):
        keys = list(dev_chunks.keys())
        return random.sample(keys, round(len(keys)*sample))

    def get_subset(d: dict, keys: list):
        return {k: d[k] for k in keys}
    sample_keys = get_sample_keys(list(dev_chunks.keys()), 0.3)
    
    # # eval_claim_embed, eval_doc_embed = eval.evaluation_preprocessing(claims_dev, claims_dev_mask, 
    # #                                                                 dev_chunks, dev_chunks_mask, model, config)
    # eval_claim_embed, eval_doc_embed = eval.evaluation_preprocessing(claims_dev, claims_dev_mask, 
    #                                                                 get_subset(dev_chunks, sample_keys), 
    #                                                                 get_subset(dev_chunks_mask, sample_keys), 
    #                                                                 model, config)
    # logger.info("FAISS retrieval")
    # precision, recall, f1, mrr = eval.retriever_score(eval_doc_embed, dev_articles_ids, eval_claim_embed, 
    #                                                 evidence_dev, labels_dev, config, k=20)
    # logger.info(f"F1: {f1}\tPrecision@{20}: {precision}\tRecall@{20}: {recall}\tMRR@{20}: {mrr}")


    logger.info("Training...")
    epoch_num = -1
    wandb.watch(model)
    for epoch_num in range(config.epoch):
        model.train()
        batch_num = len(loader)
        num_training_examples, running_loss = 0, 0.0
        for batch in tqdm(loader, total=batch_num):
            optimizer.zero_grad()
            batch = batch[0]
            num_training_examples += batch.size(0)
            if config.mode == 'finetuning':
                query, query_mask, \
                context, context_mask = util.get_finetuning_batch(batch, claims_train, claims_train_mask, evidence_train,
                                                                doc_chunks, chunks_mask, articles_ids, config)
            else:
                query, query_mask, \
                context, context_mask = util.get_pretraining_batch(ids2docs(batch, id2doc), doc_chunks, 
                                                                            tokenizer, config)

            query_cls_out = model(x=query, x_mask=query_mask)
            context_cls_out = model(x=context, x_mask=context_mask)
            logit = torch.matmul(query_cls_out, context_cls_out.transpose(-2, -1))
            correct_class = torch.tensor([i for i in range(len(query))]).long().to(config.device)
            loss = loss_fn(logit, correct_class)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch.size(0)
            epoch_avg_loss = running_loss / num_training_examples

            wandb.log({"epoch_avg_loss": epoch_avg_loss,
                       "running_loss": running_loss,
                       })

        logger.info(f"{epoch_num} epoch, train loss : {round(epoch_avg_loss, 3)}")

        # Backup save of the model
        logger.info(f"Saving backup of the model after {epoch_num+1} epochs")
        metrics.update_loss(epoch_avg_loss, epoch_num)
        if (epoch_num + 1) % 5 == 0:
            io_util.save_model(model, optimizer, metrics, f'{config.model_weight}_{epoch_num+1}')

        # Evaluation
        # if (epoch_num + 1) % 5 == 0:            
        #     logger.info("Evaluation...")
        #     # eval_claim_embed, eval_doc_embed = eval.evaluation_preprocessing(claims_dev, claims_dev_mask, 
        #     #                                                                 dev_chunks, dev_chunks_mask, model, config)

        #     eval_claim_embed, eval_doc_embed = eval.evaluation_preprocessing(claims_dev, claims_dev_mask, 
        #                                                                 get_subset(dev_chunks, sample_keys), 
        #                                                                 get_subset(dev_chunks_mask, sample_keys), 
        #                                                                 model, config)

        #     precision, recall, f1, mrr = eval.retriever_score(eval_doc_embed, dev_articles_ids, eval_claim_embed, 
        #                                                     evidence_dev, labels_dev, config, k=20)

        #     logger.info(f"F1: {f1}\tPrecision@{20}: {precision}\tRecall@{20}: {recall}\tMRR@{20}: {mrr}")
            # metrics.update_metrics(f1, precision, recall, mrr)
            # metrics.update_loss(epoch_avg_loss, epoch_num)
        #     if recall > metrics.max_rec:
        #         metrics.update_best(f1, precision, recall, mrr)
        #         io_util.save_model(model, optimizer, metrics, config.model_weight + '_best')
        #         logger.info(f"Model saved. Metrics:\n{metrics}")

    metrics.save(f'metrics/{eval.get_model_name_for_metrics(config)}')
    logger.info("Done training")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=['finetuning', 'pretraining'],
                        type=str, help="mode of work: pretraining / finetuning" + 
                                        "pretraining = pretraining on the Wikipedia data" + 
                                        "finetuning = finetuning on the Fever data")
    parser.add_argument("--task", required=True,
                        type=str, help="BFS / ICT / BFS + ICT")
    parser.add_argument("--claims_path", 
                        default="/mnt/data/factcheck/CTK/par4/ctk-data",
                        type=str, help="Path to the claims.")
    parser.add_argument("--articles_path", 
                        default="/mnt/data/factcheck/CTK/par4/interim/ctk_filtered.db",
                        type=str, help="Path to the databse with articles.")
    parser.add_argument("--articles_chunks_path", required=False, 
                        type=str, help="Path to the chunked articles data. It will speed up the things.")
                        # default="/mnt/data/factcheck/ict_chunked_data/ids-chunks-288-pretraining-ctk_filtered.pkl",
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
    # parser.add_argument("--num_steps", default=100000, type=int)
    parser.add_argument("--bs", default=32, type=int)  # 32 per GPU
    parser.add_argument("--test_bs", default=64, 
                        type=int, help="Can use a higher number than for batch_size since" +
                                        "model doesn't need to track gradients during evaluation")  # 64 per GPU
    parser.add_argument("--remove_prob", default=0.9, type=float)
    parser.add_argument("--use_cuda", type=bool, default=True)


    # parser.add_argument("--deepspeed_config", type=str, default='deepspeed_config.json')
    # parser = deepspeed.add_config_arguments(parser)  # enable to recognize DeepSpeed specific configurations

    main(parser.parse_args())
