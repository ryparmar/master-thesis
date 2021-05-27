import faiss
import numpy as np
import torch
from tqdm import tqdm
import io_util
import math
import csv

import util

class Metrics:
    def __init__(self, d=None):
        if d and isinstance(d, dict):
            self.max_pr = d['max_pr']
            self.max_rec = d['max_rec']
            self.max_f1 = d['max_f1']
            self.max_mrr = d['max_mrr']
            self.per_epoch = d['per_epoch']
        else:
            self.max_pr = -math.inf
            self.max_rec = -math.inf
            self.max_f1 = -math.inf
            self.max_mrr = -math.inf
            self.per_epoch = {
                'epoch': [],
                'loss': [],
                'pr': [],
                'rec': [],
                'f1': [],
                'mrr': []
            }

    def __str__(self):
        return (
            f"max_f1: {str(self.max_f1)}\n"
            f"max_precision: {str(self.max_pr)}\n"
            f"max_recall: {str(self.max_rec)}\n"
            f"max_mrr: {str(self.max_mrr)}\n"
            f"Per epoch:\n"
            f"  Epochs: {', '.join([str(i) for i in self.per_epoch['epoch']])}\n" 
            f"  Loss: {', '.join([str(i) for i in self.per_epoch['loss']])}\n" 
            f"  PR: {', '.join([str(i) for i in self.per_epoch['pr']])}\n" 
            f"  REC: {', '.join([str(i) for i in self.per_epoch['rec']])}\n" 
            f"  F1: {', '.join([str(i) for i in self.per_epoch['f1']])}\n" 
            f"  MRR: {', '.join([str(i) for i in self.per_epoch['mrr']])}\n")

    def print_best(self, config):
        config.logger.info(f"F1: {self.max_f1}\tPrecision: {self.max_pr}\tRecall: {self.max_rec}\tMRR: {self.max_mrr}")

    def update_loss(self, loss, epoch):
        self.per_epoch['epoch'].append(epoch)
        self.per_epoch['loss'].append(loss)

    def update_metrics(self, f1, pr, rec, mrr):
        self.per_epoch['f1'].append(f1)
        self.per_epoch['rec'].append(rec)
        self.per_epoch['pr'].append(pr)
        self.per_epoch['mrr'].append(mrr)

    def update_best(self, f1, pr, rec, mrr):
        self.max_f1 = f1
        self.max_pr = pr
        self.max_rec = rec
        self.max_mrr = mrr

    def save(self, path, as_str=True):
        with open(path, 'w') as fw:
            if as_str:
                fw.write(str(self))
            else:
                writer = csv.writer(fw, delimiter=';', quoting=csv.QUOTE_MINIMAL)
                writer.writerow(['epoch', 'loss', 'f1', 'pr', 'rec', 'mrr'])
                for e in range(len(self.per_epoch['epoch'])):
                    writer.writerow([self.per_epoch['epoch'][e], self.per_epoch['loss'][e],
                                    self.per_epoch['f1'], self.per_epoch['pr'], self.per_epoch['rec'],
                                    self.per_epoch['mrr']])


# Main Eval Caller Function
# def evaluate(claims_dev, claims_dev_mask, dev_chunks, dev_chunks_mask, 
#              evidence_dev, labels_dev, dev_articles_ids, 
#              metrics, model, optimizer, config, 
#              k=[10], save_model=False, save_path=config.model_weight):
#     eval_claim_embeddings, \
#     eval_document_embeddings = evaluation_preprocessing(claims_dev, claims_dev_mask, dev_chunks, dev_chunks_mask, 
#                                                              model, config)

#     for kk in k:
#         precision, recall, f1, mrr = retriever_score(eval_claim_embeddings, eval_document_embeddings, 
#                                                     evidence_dev, labels_dev, dev_articles_ids, config, k=kk)
#         config.logger.info(f"F1: {f1}\tPrecision@{kk}: {precision}\tRecall@{kk}: {recall}\tMRR@{kk}: {mrr}")
#         metrics.update_metrics(f1, precision, recall, mrr)
#         if recall > metrics.max_rec:
#             metrics.update_best(f1, precision, recall, mrr)
#             if save_model:
#                 io_util.save_model(model, optimizer, save_path)


def search_top_k(corp_emb, query_emb, embedding_dim, k, config):
    """
    Returns a tuple with ordered lists of lists of cosine distances between and top k matches in corpus_embeddings.
    Each list corresponds to one query.
    Needs GPU
    Available metrics = faiss.METRIC_INNER_PRODUCT, faiss.METRIC_L2, ...
    more here https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances

    return type = (List[List[int = cosine_distance]], List[List[int = index_of_corpus_embedding]]);
    type(cosine_distance) == Float
    type(index_of_corpus_embedding) == Int
    """
    config.logger.info(f"Preparing index and executing the search with embedding dim {embedding_dim}")
    index = faiss.index_factory(embedding_dim, "PCA384,Flat", faiss.METRIC_INNER_PRODUCT)  # Flat = exhaustive search
    faiss.normalize_L2(corp_emb)  # need to normalize query and corpus vectors for cosine distance
    faiss.normalize_L2(query_emb)
    if config.device != 'cpu':
        res = faiss.StandardGpuResources()
        if len(config.devices) > 1:
            dev_index = faiss.index_cpu_to_all_gpus(index)  # use gpu
        else:
            dev_index = faiss.index_cpu_to_gpu(res, 0, index)
    else:
        dev_index = index
    
    dev_index.train(corp_emb)
    dev_index.add(corp_emb)
    return dev_index.search(query_emb, k)  # return distances, indices matrices


def evidence_macro_precision(evidence, label, predicted_evid, max_evidence=None, page_only=True):
    """
    precision = predicted
    """
    this_precision = 0.0
    this_precision_hits = 0.0

    if label.upper() != "NOT ENOUGH INFO":
        if isinstance(evidence, list):
            if page_only:
                all_evi = [e[2] for eg in evidence for e in eg if e[3] is not None]
            else:
                all_evi = [[e[2], e[3]] for eg in evidence for e in eg if e[3] is not None]
        elif isinstance(evidence, str):
            all_evi = [evidence]
        else:
            print("UNEXPECTED EVIDENCE TYPE! in evidence_macro_precisin function!")

        for prediction in predicted_evid:
            if prediction in all_evi:
                this_precision += 1.0
            this_precision_hits += 1.0

        return (this_precision / this_precision_hits) if this_precision_hits > 0 else 1.0, 1.0

    return 0.0, 0.0


def evidence_macro_recall(evidence, label, predicted_evidence, max_evidence=None, page_only=True):
    # We only want to score F1/Precision/Recall of recalled evidence for NEI claims
    if label.upper() != "NOT ENOUGH INFO":
        # If there's no evidence to predict, return 1
        if len(evidence) == 0:  # or all([len(eg) == 0 for eg in instance]):
            return 1.0, 1.0

        if isinstance(evidence, list):
            for evidence_group in evidence:
                evidence = [e[2] for e in evidence_group] if page_only else [[e[2], e[3]] for e in evidence_group]
                # We only want to score complete groups of evidence. Incomplete groups are worthless.
                if all([item in predicted_evidence for item in evidence]):
                    return 1.0, 1.0
        elif isinstance(evidence, str):
            return 1.0, 1.0 if evidence in predicted_evidence else 0.0, 1.0

        return 0.0, 1.0
    return 0.0, 0.0

# TODO prepsat z doc-retr od honzy. otestovat, jestli se to nepocita spatne?!
def evidence_macro_mrr(evidence, label, predicted_evidence, max_evidence=None, page_only=True):
    """Return Mean Reciprocal Rank"""
    if label.upper() != "NOT ENOUGH INFO":
        # If there's no evidence to predict, return 1
        if len(evidence) == 0 or all([len(eg) == 0 for eg in evidence]):
            return 1.0, 1.0

        predicted_evidence = predicted_evidence if page_only else None
        # at most max_evidence predicted evidence elements
        predicted_evidence = predicted_evidence if max_evidence is None else predicted_evidence[:max_evidence]

        mrr = []
        if isinstance(evidence[0], list):
            for evidence_group in evidence:
                if page_only:
                    evidence = [e[2] for e in evidence_group]
                    # include only the worst part of document group
                    # e.g. 2 relevant documents (a, b) - a is 2nd in predicted and b is 9th in predicted -> mrr = 1/9
                    ranks = [np.where(predicted_evidence == ev)[0][-1] for ev in evidence if ev in predicted_evidence]
                    if len(ranks) == len(evidence):
                        mrr.append(1 / (max(ranks) + 1))
                    else:
                        mrr.append(0)
                else:
                    evidence = [[e[2], e[3]] for evidence_group in evidence for e in evidence_group]
                    mrr.append(0)  # TODO
        elif isinstance(evidence[0], str):
            if page_only:
                rank = np.where(predicted_evidence == evidence)[0][-1] if evidence in predicted_evidence else None
                if rank:
                    mrr.append(1 / (max(ranks) + 1))
                else:
                    mrr.append(0)
        return sum(mrr), len(mrr)
    return 0.0, 0.0


def retriever_score(corpus_embeddings, corpus_ids, query_embeddings, evidence, labels, config, k=20):
    macro_precision, macro_precision_hits = 0, 0
    macro_recall, macro_recall_hits = 0, 0
    macro_mrr, macro_mrr_hits = 0, 0

    D, I = search_top_k(corpus_embeddings, np.asarray(query_embeddings), corpus_embeddings.shape[-1], k, config)
    # config.logger.info(f"D shape: {D.shape}\tI shape: {I.shape}")

    for i, top_k_idxs in tqdm(enumerate(I), desc='Calculating evaluation metrics'):
        predicted_evidence = np.take(corpus_ids, I[i])

        macro_prec = evidence_macro_precision(evidence[i], labels[i], predicted_evidence)
        macro_precision += macro_prec[0]
        macro_precision_hits += macro_prec[1]

        macro_rec = evidence_macro_recall(evidence[i], labels[i], predicted_evidence)
        macro_recall += macro_rec[0]
        macro_recall_hits += macro_rec[1]

        macro_rr = evidence_macro_mrr(evidence[i], labels[i], predicted_evidence)
        macro_mrr += macro_rr[0]
        macro_mrr_hits += macro_rr[1]

    pr = np.round((macro_precision / macro_precision_hits), 6) if macro_precision_hits > 0 else 1.0
    rec = np.round((macro_recall / macro_recall_hits), 6) if macro_recall_hits > 0 else 0.0
    mrr = np.round((macro_mrr / macro_mrr_hits), 6) if macro_mrr_hits > 0 else 0.0
    f1 = np.round((2.0 * pr * rec / (pr + rec + 1e-6)), 6)
    return pr, rec, f1, mrr


def encode_chunks(chunks, chunks_mask, model, batch_size):
    chunks_encode = []
    with torch.no_grad():
        for i in tqdm(range(0, chunks.shape[0], batch_size), desc='Embedding given chunks...'):
            cls_output = model(x=chunks[i:i + batch_size],
                            x_mask=chunks_mask[i:i + batch_size])
            chunks_encode.append(cls_output.detach().cpu().numpy())
    embeddings = np.concatenate(chunks_encode, axis=0)
    return embeddings

# TODO UGLY AS FUCK
def encode(doc_chunks, chunks_mask, model, batch_size=32):
    embeddings, doc_ids, par_ids = [], [], []
    chunks, masks = doc_chunks, chunks_mask
    if isinstance(doc_chunks, dict) and isinstance(chunks_mask, dict):
        chunks, masks = [], []
        for doc_id, doc in tqdm(doc_chunks.items(), desc='Generating chunks embeddings...'):
            doc_ids.append(doc_id)
            if isinstance(doc, list):
                chunks.append(util.flatten_list(doc))
                masks.append(util.flatten_list(chunks_mask[doc_id]))
            elif isinstance(doc, dict):
                for par_id, par in doc.items():
                    par_ids.append(par_id)
                    if isinstance(par, list):
                        chunks.append(util.flatten_list(par))
                        masks.append(util.flatten_list(chunks_mask[doc_id][par_id]))
                    else:
                        chunks.append(torch.flatten(par))
                        masks.append(torch.flatten(chunks_mask[doc_id][par_id]))
            else:
                chunks.append(torch.flatten(doc))
                masks.append(torch.flatten(chunks_mask[doc_id]))

        #elif isinstance(doc_chunks, list) and isinstance(masks, list):
        chunks = torch.stack(chunks, axis=0)
        masks = torch.stack(masks, axis=0)
    
    embeddings = encode_chunks(chunks, masks, model, batch_size)
    # print(f"Shape of embeddings {embeddings.shape}")
    return embeddings


def evaluation_preprocessing(claims_tokenized, claim_masks, context_chunks, context_chunk_masks, model, config):
    model.eval()
    document_embeddings = encode(context_chunks, context_chunk_masks, model, batch_size=config.test_bs)
    config.logger.info(f"Documents embedded with shape: {document_embeddings.shape}")
    claim_embeddings = encode(claims_tokenized, claim_masks, model, batch_size=config.test_bs)
    config.logger.info(f"Claims embedded with shape: {claim_embeddings.shape}")
    return claim_embeddings, document_embeddings


def get_model_name_for_metrics(config):
    name = f"{config.bert_model}_{config.task}_{config.mode}_{str(config.max_seq)}_{str(config.epoch)}ep_{config.date}"
    return name