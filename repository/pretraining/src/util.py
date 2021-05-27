import re, os
import nltk
from tqdm import tqdm
import numpy as np
import torch
import random
import itertools

import io_util


def make_mask(x, pad_idx, decode=False):
    """Create a mask to hide padding and future words."""
    mask = (x != pad_idx)
    if decode:
        size = x.shape[-1]
        attn_shape = (1, size, size)
        subsequent_mask = np.triu(np.ones(attn_shape), k=1)
        mask = np.expand_dims(mask, axis=1) & (subsequent_mask == 0)
    return mask.astype('uint8')

# CTK paragraph_id is composed of document_id and paragraph_number
def split_par_id(doc_id: str) -> (str, str):
    """Returns doc_id, par_id"""
    tmp = doc_id.split('_')
    assert len(tmp) == 2
    return tmp[0], tmp[1] 

# def flatten(list2d):
#     return list(itertools.chain.from_iterable(list2d))

def flatten(x):
    if isinstance(x, list):
        return [i for i in flatten_list(x)]
    elif isinstance(x, dict):
        return [i for i in flatten_dict(x)]  # chunks
    print("UNSUPPORTED OPERATION -- INPUT MUST BE DICT OR LIST! flattne function")
    return None

# Flatten arbitrary nested lists
def flatten_list(container):
    for i in container:
        if isinstance(i, (list)):
            for j in flatten(i):
                if j:
                    yield j
        else:
            if i:
                yield i

def flatten_dict(d):
    for k, v in d.items():
        if isinstance(v, (dict)):
            for kk, vv in flatten_dict(v):
                if vv:
                    yield vv
        else:
            if v:
                yield v

def pad_chunk(chunk: list, config, add_cls_token=False, get_mask=True, decode=False, pad_max=True, device='cpu'):
    max_seq = config.max_seq if add_cls_token else config.max_seq - 1
    pad_seq = np.ones((max_seq), dtype=np.int64) * config.pad_token_id
    # pad_seq.fill(config.pad_token_id)
    pad_seq[0] = config.cls_token_id
    pad_seq[1:len(chunk)+1] = np.array(chunk)

    mask = None
    if get_mask:
        mask = make_mask(pad_seq, config.pad_token_id, decode)
        mask = torch.from_numpy(mask).to(device)
    return torch.from_numpy(pad_seq).to(device), mask

def pad_chunks(chunks, config, add_cls_token=False, get_mask=True, decode=False, pad_max=True, device="cpu"):
    """
    padding given sequence with maximum length 
    generate padded sequence and mask
    """
    max_seq = config.max_seq if add_cls_token else config.max_seq - 1
    if isinstance(chunks, list):
        seq_len = np.array([min(len(chunk), max_seq) for chunk in chunks])
    elif isinstance(chunks, dict):
        seq_len = np.array([min(len(chunk), max_seq) for _, chunk in chunks.items()])  # par = max 1 chunk (first 288 tokens)
    else:
        print("Wrong input type when padding chunks!")

    if not pad_max:
        max_seq = max(seq_len)

    pad_seq = np.zeros((len(chunks), max_seq), dtype=np.int64)
    pad_seq.fill(config.pad_token_id)

    if isinstance(chunks, list):
        for i, chunk in enumerate(chunks):
            if add_cls_token:
                pad_seq[i, 0] = config.cls_token_id
                pad_seq[i, 1:seq_len[i]+1] = chunk[:seq_len[i]]
            else:
                pad_seq[i, :seq_len[i]] = chunk[:seq_len[i]]
    elif isinstance(chunks, dict):
        for i, (_, chunk) in enumerate(chunks.items()):
            if add_cls_token:
                pad_seq[i, 0] = config.cls_token_id
                pad_seq[i, 1:seq_len[i]+1] = chunk[:seq_len[i]]
            else:
                pad_seq[i, :seq_len[i]] = chunk[:seq_len[i]]

    if get_mask:
        mask = make_mask(pad_seq, config.pad_token_id, decode)
        mask = torch.from_numpy(mask).to(device)
    else:
        mask = None
    return torch.from_numpy(pad_seq).to(device), mask


def tokenize_sentences(sentences: list, tokenizer) -> list:
    """Split paragraphs into sentences and tokenize them."""
    sentences_tokenized = [tokenizer.encode(sentence, add_special_tokens=False) for sentence in sentences]  # tokenize 
    return sentences_tokenized


def paragraph_to_sentences(documents):
    """Split paragraphs into sentences."""
    if isinstance(documents, dict):
        doc2sentences = {doc_id: [nltk.sent_tokenize(par) for par in doc] 
                            for doc_id, doc in tqdm(documents.items(), desc='Splitting into sentences')}
    else:
        doc2sentences = [nltk.sent_tokenize(par) for par in documents]
    return doc2sentences


def tokenize_documents(documents: dict, tokenizer) -> dict:
    """Tokenize documents per sentence."""
    # if doc_ids:
    #     docs_tokenized = {doc_id: [tokenize_sentences(par, tokenizer) 
    #                                 for par in paragraphs_to_sentences(documents[doc_id])] 
    #                     for doc_id in tqdm(doc_ids, desc='Tokenizing documents')}
    # else:
    docs_tokenized = {
        doc_id: {
            par_id: tokenize_sentences(nltk.sent_tokenize(par), tokenizer) 
            for par_id, par in doc.items()
        }         
        for doc_id, doc in tqdm(documents.items(), desc='Tokenizing documents')
    }
    return docs_tokenized


def nested_list_len(lst: list) -> int:
    """Calculates the number of tokens in a nested list"""
    total_sum = 0
    for sub_lst in lst:
        total_sum += len(sub_lst)
    return total_sum


def create_chunks(data: dict, tokenizer, config, as_eval=False):
    """
    Creates chunks of text with given maximal length (max_seq).
    Data argument must have document-paragraphs-sentences structure
    
    Input shape:
        Dict[Dict[List[str]]]

    Return shape: 
        Dict[List[]] -- list of chunks for each document in case of pretraining
        Dict[Dict[List[]]] -- list of chunks for each document and paragraph in case of finetuning

    Pretraining chunks are created on a level of document = in a chunk can be sentences from two or more different 
    paragraphs that is the reason of merging CTK data into doc data. This will result in more than chunks than documents.

    Finetuning chunks are created on a level of paragraph = in a chunk can be sentences only from a single paragraph.
    Chunks is created by filling it with first max_seq tokens from a paragraph and rest is thrown away.
    This will result in equivalent number of chunks and paragraphs (documents in case doc = single paragraph).
    """
    doc_chunks = {}
    # has to be the same as the max sequence length your model was trained on; -1 for CLS toke
    max_len = config.max_seq - 1
    if config.mode == 'pretraining' and not as_eval: 
        for doc_id, doc in tqdm(data.items(), desc='Creating chunks'):
            chunk, chunks = [], []
            for par_id, par in doc.items():
                for sentence in par:
                    if nested_list_len(chunk) + len(sentence) > max_len:
                        chunks.append(chunk)
                        chunk = []  # list of sentences in a chunk
                    chunk.append(sentence)
            if len(chunk) > 1:  # keep only chunks with more than one sentence
                chunks.append(chunk)
            doc_chunks[doc_id] = chunks
    elif config.mode == 'finetuning':
        for doc_id, doc in tqdm(data.items(), desc='Creating chunks'):
            doc_chunks[doc_id] = {}
            for par_id, par in doc.items():
                chunk = []
                for sentence in par:
                    chunk += sentence
                try:
                    doc_chunks[doc_id][int(par_id)] = chunk[:max_len]
                except:
                    doc_chunks[doc_id][int(par_id)] = chunk[:max_len]
    return doc_chunks


def remove_invalid_pars(pars, par_ids):
    """Removes paragraphs which doesnt end with '.' or ends with '...'"""
    ret_pars, ret_par_ids = [], []
    for pid, p in tqdm(zip(par_ids, pars)):
        if not p.strip().endswith('...') and p.strip().endswith('.'):
            ret_pars.append(p)
            ret_par_ids.append(pid)
    return ret_pars, ret_par_ids


def transform_ctk(pars: list, par_ids: list):
    """
    Transform paragraphs, paragraph_ids into dicts of article: paragraphs and article_id: paragraph_ids
    This is done in order to have an unified structure for all the data: document-paragraph-sentence in a dictionary of 
    articles.
    Used for CTK data which have paragraph granularity in default.
    """
    docs = {}
    for par, idx in tqdm(zip(pars, par_ids), desc='Transforming data'):
        doc_title, par_num = idx.split('_')
        if doc_title in docs:
            docs[doc_title][par_num] = par
        else:
            docs[doc_title] = {par_num: par}
    return docs


def transform_wiki(data: list):
    """
    Transform paragraphs, paragraph_ids into dicts of article: paragraphs and article_id: paragraph_ids
    This is done in order to have an unified structure for all the data: document-paragraph-sentence in a dictionary of 
    articles.
    Used for WIKI data with articles in full length.
    """
    docs = {}
    for doc in tqdm(data, desc='Transforming data'):
        title = doc['title'].strip()
        if title != 'Hlav,í strana':
            for par_num, par in enumerate(doc['text'].split('\n\n')):
                if title in docs:
                    docs[title][par_num] = par.strip()
                else:
                    docs[title] = {par_num: par.strip()}
    return docs


def transform_fever_wiki(data: list, data_ids: list):
    """
    Transform data into dictionary of articles: paragraphs
    This is done in order to have an unified structure for all the data: document-paragraph-sentence in a dictionary of
    articles.
    Used for FEVER db data with articles having only the first (abstract) paragraph.
    """
    docs = {}
    for doc, doc_id in tqdm(zip(data, data_ids), desc='Transforming data'):
        if doc_id in docs:
            key = len(docs[doc_id])
            docs[doc_id][key] = doc.strip()
        else:
            docs[doc_id] = { 0: doc.strip() }  # doc = single paragraph; due to doc-par-sent structure
    return docs


def process_wiki(path: str, tokenizer, config):
    """Tokenize and create chunks of WIKI data."""
    filename = path.split('/')[-1]
    filetype = filename.split('.')[-1]
    config.logger.info(f"Processing WIKI data from {path} as a {filetype}.")
    if filetype == 'json':
        wiki_json = io_util.load_json(path)
        docs = transform_wiki(wiki_json)
        docs_tokenized = tokenize_documents(docs, tokenizer)
        chunks = create_chunks(docs_tokenized, tokenizer, config)
        config.logger.info(f"WIKI data chunks prepared.")
    elif filetype == 'db':
        pass  # TODO
    else:
        config.logger.error(f"Unexpected filetype {filetype} in {process_wiki.__name__}.")
    return chunks


def batches(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def process_ctk(path: str, tokenizer, config):
    """Tokenize and create chunks of CTK data."""
    filename = path.split('/')[-1]
    filetype = filename.split('.')[-1]
    config.logger.info(f"Processing CTK data from {path} as a {filetype}. This takes quite some time.")
    if filetype == 'db':
        pars, par_ids = io_util.load_db(path)  # returns paragraphs and paragraph ids
        if config.mode == 'pretraining':
            pars, par_ids = remove_invalid_pars(pars, par_ids)
        docs = transform_ctk(pars, par_ids)  # group paragraphs for each article
        del pars
        
        # generate at once
        docs_tokenized = tokenize_documents(docs, tokenizer)  # split to sentences and tokenize
        doc_chunks = create_chunks(docs_tokenized, tokenizer, config)  # chunking
        config.logger.info(f"CTK data chunks prepared.")
    else:
        config.logger.error(f"Unexpected filetype {filetype} in {process_ctk.__name__}.")
    return doc_chunks


def process_fever_wiki(path: str, tokenizer, config, as_eval=False):
    """Tokenize and create chunks of FEVER wiki data. Contains only the first paragraph (abstract) of the  WIKI data."""
    filename = path.split('/')[-1]
    filetype = filename.split('.')[-1]
    config.logger.info(f"Processing FEVER wiki data from {path} as a {filetype}.")
    if filetype == 'db':
        pars, par_ids = io_util.load_db(path)  # returns paragraphs and paragraph ids
        docs = transform_fever_wiki(pars, par_ids)  # group paragraphs of articles
        docs_tokenized = tokenize_documents(docs, tokenizer)
        doc_chunks = create_chunks(docs_tokenized, tokenizer, config, as_eval)
        config.logger.info(f"FEVER wiki data chunks prepared.")
    else:
        config.logger.error(f"Unexpected filetype {filetype} in {process_fever_wiki.__name__}.")
    return doc_chunks


def make_chunks(path: str, tokenizer, config, as_eval=False, save_chunks=False):
    """
    Converts file to a list of tokenized chunks.
    Used for unsupervised pretraining.
    """
    filename = path.split('/')[-1].split('.')[0]
    if as_eval:
        saved_filename = f"/mnt/data/factcheck/ict_chunked_data/ids-chunks-{config.max_seq}-finetuning-{filename}.pkl"
    elif config.articles_chunks_path:
        saved_filename = config.articles_chunks_path
    else:
        saved_filename = f"/mnt/data/factcheck/ict_chunked_data/ids-chunks-{config.max_seq}-{config.mode}-{filename}.pkl"

    doc_chunks = io_util.load_doc_chunks(saved_filename)
    if doc_chunks:
        doc_chunks = doc_chunks[0]
        config.logger.info(f"Chunks of {len(doc_chunks)} documents loaded from {saved_filename}.")
        # if config.task.upper() == 'ICT' and config.mode == 'pretraining': 
        #     doc_chunks = flatten_chunks(doc_chunks)
        return doc_chunks

    if 'wiki' in filename:
        doc_chunks = process_wiki(path, tokenizer, config)
    elif 'ctk' in filename:
        doc_chunks = process_ctk(path, tokenizer, config)
    elif 'fever' in filename:
        doc_chunks = process_fever_wiki(path, tokenizer, config)
    
    if save_chunks:
        io_util.save_doc_chunks(saved_filename, doc_chunks)
        config.logger.info(f"Chunks saved in {saved_filename}.")

    # if config.task.upper() == 'ICT' and config.mode == 'pretraining': 
    #     doc_chunks = flatten_chunks(doc_chunks)
    return doc_chunks


def get_par_ids(doc_chunks: dict) -> list:
    return [f"{doc_id}_{par_id}" for doc_id, doc in doc_chunks.items() for par_id, _ in doc.items()]


def get_num_chunks(doc_chunks: dict) -> list:
    return sum([len(doc) for _, doc in doc_chunks.items()])


def process_chunks(doc_chunks, config):
    """Add CLS token in each chunk, pad and create masks."""
    assert isinstance(doc_chunks, dict) 
    ret_chunks, ret_masks = {}, {}
    for doc_id, doc in tqdm(doc_chunks.items(), desc='Padding chunks...'):
        if isinstance(doc, dict):  # finetuning
            chunk, mask = {}, {}
            for par_id, par in doc.items():
                ch, m = pad_chunk(flatten(par), config, add_cls_token=True)
                chunk[par_id] = ch
                mask[par_id] = m
        else:
            chunk, mask = pad_chunks(doc, config, add_cls_token=True)
        ret_chunks[doc_id] = chunk
        ret_masks[doc_id] = mask
    return ret_chunks, ret_masks

def load_claims(split: str, config, path=None):
    """
    split = ["train", "dev", "test"]
    :return: (claims, evidence, labels)
    """
    if split in ['train', 'dev', 'test']:
        split_path = os.path.join(config.claims_path, split +'.jsonl') if not path else path
        claims_data = io_util.load_jsonl(split_path)
        db_cols = ['id', 'verifiable', 'label', 'claim', 'evidence', 'claim_en']

        data = {}
        for c in claims_data:
            for col in db_cols:
                if col in data:
                    data[col].append(c.get(col, float('nan')))
                else:
                    data[col] = [c.get(col, float('nan'))]

        assert len(data['claim']) == len(data['evidence']) == len(data['label'])
        print(f"Loaded {len(data['claim'])} claims from {split} split.")
        config.logger.info(f"Loaded {len(data['claim'])} claims from {split} split.")
        return data['claim'], data['evidence'], data['label']
    else:
        config.logger.error(f"Inputed {split} split is not valid! Valids are: train / dev / test only.")


def process_claims(claims, tokenizer, config, _pad_max=True):
    """Tokenizes, pads and creates masks for given claims."""
    tokenized_claims = tokenize_sentences(claims, tokenizer)
    # [claim.insert(0, config.cls_token_id) for claim in tokenized_claims]  # inserts [CLS] token at the beginning
    claims, claim_masks = pad_chunks(tokenized_claims, config, add_cls_token=True, pad_max=_pad_max, device=config.device)
    return claims, claim_masks


def remove_unverifiable_claims(claims, evidence, labels, config):
    """
    Removes unverifiable claims and split the claims having more than a single evidence document into more claims.
    Relation 1 claim = 1 evidence doc.

    Example:
    claim A having 2 evidence documents is splitted into two:
    claim A - evidence doc 1; claim A - evidence doc 2;
    """
    claims_processed, evidence_processed, labels_processed = [], [], []
    for evid, claim, label in zip(evidence, claims, labels):
        # turns fever evidence set into a list of all unique document titles this claim is verifiable by
        all_evi = list(set([e[2] for eg in evid for e in eg if e[3] is not None]))
        if all_evi:
            # unrolling claims adds ~ extra claims. so some claims may be overrepresented
            for evi in all_evi:
                claims_processed.append(claim)
                evidence_processed.append(evi)
                labels_processed.append(label)

    assert len(claims_processed) == len(evidence_processed) == len(labels_processed)
    config.logger.info(f"{len(claims_processed)} claims returned.")  #len(claims) - 
    return claims_processed, evidence_processed, labels_processed


def remove_invalid_claims(claims, evidence, labels, doc_titles, config):
    """
    Removes claims with evidence which doesn't match any document. 
    This usually happens when the wikipedia article was renamed.
    """
    # if the documents in the evidence is missing - particular claim, label and evidence is deleted
    assert len(evidence) == len(claims) == len(labels)
    new_evidence, new_claims, new_labels = [], [], []

    for i, doc_id in tqdm(enumerate(evidence), desc='Validating evidences'):
        if doc_id in doc_titles:
            new_evidence.append(evidence[i])
            new_claims.append(claims[i])
            new_labels.append(labels[i])
    config.logger.info(f"{len(new_claims)} claims returned.")  #removed due to missing evidence document.")  #len(claims) - 
    return new_claims, new_evidence, new_labels

# def ict_pretraining_targets_and_contexts(index, chunks, config):
#     """
#     Randomly splits chunk into target sentence and context sentences.
#     ICT - in a single batch there can be multiple contexts (pars) from a single document/article
#     ICT = query - paragraph relation
#     """
#     sentences = [chunks[i] for i in index if chunks[i]]  # pick the chunks in a batch and removes []
#     target_sentence_id = [random.randint(0, len(sen) - 1) for sen in sentences] # randomly choose target sentence - claim
#     # randomly choose whether to remove the target sentence from a chunk
#     remove_target = [random.random() < config.remove_prob for _ in
#                     range(len(target_sentence_id))]

#     target_context = [sen[:i] + sen[i + remove:] for i, sen, remove in
#                      zip(target_sentence_id, sentences, remove_target)]
#     target_context = [[y for x in context for y in x] for context in target_context]
#     target_context = [[config.cls_token_id] + context for context in target_context]

#     target_sentence = [sen[i] for i, sen in zip(target_sentence_id, sentences)]
#     target_sentence = [[config.cls_token_id] + sen for sen in target_sentence]
#     return target_context, target_sentence


def get_rand_chunk_id(chunks: list):
    """Returns a random chunk id from a list of chunks."""
    if len(chunks) > 0:
        return random.sample(chunks, 1)[0]
    else:
        print("ERROR -- EMPTY CHUNKS ON INPUT!")

def get_diff_rand_chunk_ids(chunks: list):
    """Returns a two random chunk ids from a list of chunks."""
    if len(chunks) > 1:
        rand_sentence_chunk_id, rand_context_chunk_id = random.sample(range(len(chunks)), 2)
        return rand_sentence_chunk_id, rand_context_chunk_id
    elif len(chunks) > 0:
        return 0, 0
    else:
        print("ERROR -- EMPTY CHUNKS ON INPUT!")


def remove_sentence_from_context(sen_id, con):
    assert isinstance(sen_id, int)
    assert isinstance(con, list)
    return con[:sen_id] + con[sen_id + 1:]


def bfs_pretraining_targets_and_contexts(batch_ids, doc_chunks, config):
    """
    Randomly splits chunk into target sentence and context sentences.
    BFS - in a single batch there cannot be more contexts (pars) from a single document/article
    BFS = query - document relation
    """
    # print("BFS batch")
    target_sentence, target_context = [], []
    for doc_id in batch_ids:
        if doc_chunks[doc_id]:
            sen_chunk_id, con_chunk_id = get_diff_rand_chunk_ids(doc_chunks[doc_id])
            sen_id = (random.randint(0, len(doc_chunks[doc_id][sen_chunk_id]) - 1) 
                        if len(doc_chunks[doc_id][sen_chunk_id]) > 1 else 0)

            if len(doc_chunks[doc_id][sen_chunk_id]) > 0:  # non-empty chunk
                target_sentence.append([config.cls_token_id] + doc_chunks[doc_id][sen_chunk_id][sen_id])

                # if there is an only single chunk, return rand sentence and chunk wo that sentece
                if sen_chunk_id == con_chunk_id:
                    target_context.append([config.cls_token_id] + 
                        flatten(
                            remove_sentence_from_context(sen_id, doc_chunks[doc_id][con_chunk_id])))
                else:
                    target_context.append([config.cls_token_id] + flatten(doc_chunks[doc_id][con_chunk_id]))
    return target_context, target_sentence


def ict_pretraining_targets_and_contexts(batch_ids, doc_chunks, config):
    """
    Randomly splits chunk into target sentence and context sentences.
    ICT - in a single batch there can be multiple contexts (pars) from a single document/article
    ICT = query - paragraph relation
    """
    # print("ICT batch")
    target_sentence, target_context = [], []
    for doc_id in batch_ids:
        if doc_chunks[doc_id]:
            chunk_id = get_rand_chunk_id(range(len(doc_chunks[doc_id])))
            sen_id = (random.randint(0, len(doc_chunks[doc_id][chunk_id]) - 1) 
                        if len(doc_chunks[doc_id][chunk_id]) > 1 else 0)
            if len(doc_chunks[doc_id][chunk_id]) > 0:  # non-empty chunk
                target_sentence.append([config.cls_token_id] + doc_chunks[doc_id][chunk_id][sen_id])

                remove_target_sen = random.random() < config.remove_prob
                # if there is an only single chunk, return rand sentence and chunk wo that sentece
                if remove_target_sen:
                    target_context.append([config.cls_token_id] + 
                        flatten(
                            remove_sentence_from_context(sen_id, doc_chunks[doc_id][chunk_id])))
                else:
                    target_context.append([config.cls_token_id] + flatten(doc_chunks[doc_id][chunk_id]))
    return target_context, target_sentence


def get_pretraining_batch(batch_ids, chunks, tokenizer, config):
    """
    Returns batch for ICT pretraining.
    Chunk (senteces of text with max length = max_seq parameters in config) splitted into target sentece and context.
    """
    if config.task.upper() == 'BFS':
        # chunks here are on the level of documents: List[List[List[tokens]]] -- Documents-Chunks-Senteces-Tokens
        target_context, target_sentence = bfs_pretraining_targets_and_contexts(batch_ids, chunks, config)
    elif config.task.upper() == 'ICT':
        # chunks here are missing Documet level info: List[List[tokens]] -- Chunks-Senteces-Tokens
        target_context, target_sentence = ict_pretraining_targets_and_contexts(batch_ids, chunks, config)
    else:
        bfs = random.randint(0, 1)  # randomly choose between bfs / ict
        target_context, target_sentence = (bfs_pretraining_targets_and_contexts(batch_ids, chunks, config) if bfs 
                                           else ict_pretraining_targets_and_contexts(batch_ids, chunks, config))

    target, target_mask = pad_chunks(target_sentence, config, device=config.device)
    context, context_mask = pad_chunks(target_context, config, device=config.device)
    return target, target_mask, context, context_mask


def get_rand_chunk(chunks, idx):
    """Returns a random chunk from a document."""
    rand_chunk_id = random.randint(0, len(chunks)-1)
    tries = 0
    while not chunks[rand_chunk_id] or rand_chunk_id == idx:
        rand_chunk_id = random.randint(0, len(chunks)-1)
        if tries == 10 and not chunks[0]:
            return []
        else:
            return 0
    return rand_chunk_id


def bfs_finetuning_contexts(doc_chunks, doc_chunks_mask, evidence, is_ctk):
    docs_pad, docs_mask = [], []
    for ev_id in evidence:
        if is_ctk:
            doc_id, chunk_id = split_par_id(ev_id) 
        else:
            ev_id, 0
        rand_chunk_id = get_rand_chunk_id(list(doc_chunks[doc_id].keys()))
        docs_pad.append(doc_chunks[doc_id][rand_chunk_id])
        docs_mask.append(doc_chunks_mask[doc_id][rand_chunk_id])
    return docs_pad, docs_mask


def ict_finetuning_contexts(doc_chunks, doc_chunks_mask, evidence, is_ctk):
    docs_pad, docs_mask = [], []
    for ev_id in evidence:
        if is_ctk:
            doc_id, chunk_id = split_par_id(ev_id) 
        else:
            ev_id, 0
        docs_pad.append(doc_chunks[doc_id][int(chunk_id)])
        docs_mask.append(doc_chunks_mask[doc_id][int(chunk_id)])
    return docs_pad, docs_mask

def get_finetuning_batch(batch_ids, claims, claims_mask, evidence, doc_chunks, doc_chunks_mask, doc_ids, config):
    """
        Note: some negatives could be positives, as some claims have multiple supporting documents,
        but that's being ignored here for convenience.
    """
    claims_pad = [claims[i] for i in batch_ids]
    claims_mask = [claims_mask[i] for i in batch_ids]
    evidence = [evidence[i] for i in batch_ids]  # document ids
    # doc_batch_ids = [doc_ids[i] for i in batch_ids]

    # get documents
    # 1 evidence = 1 doc id, see remove_unverifiable_claims
    is_ctk = 'CTK' in config.articles_path
    if config.task.upper() == 'BFS':  # works only for CTK data
        docs_pad, docs_mask = bfs_finetuning_contexts(doc_chunks, doc_chunks_mask, evidence, is_ctk)
    elif config.task.upper() == 'ICT':
        docs_pad, docs_mask = ict_finetuning_contexts(doc_chunks, doc_chunks_mask, evidence, is_ctk)
    else:
        bfs = random.randint(0, 1)  # randomly choose between bfs / ict
        docs_pad, docs_mask = (bfs_finetuning_contexts(doc_chunks, doc_chunks_mask, evidence, is_ctk)
                               if bfs else 
                               ict_finetuning_contexts(doc_chunks, doc_chunks_mask, evidence, is_ctk))

    c_p = torch.stack(claims_pad, axis=0).to(torch.int64).to(config.device)
    c_m = torch.stack(claims_mask, axis=0).to(torch.int64).to(config.device)
    d_p = torch.stack(docs_pad, axis=0).to(torch.int64).to(config.device)
    d_m = torch.stack(docs_mask, axis=0).to(torch.int64).to(config.device)
    return c_p, c_m, d_p, d_m