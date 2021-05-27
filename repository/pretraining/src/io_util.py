import torch
import pickle
import csv
import json
import sqlite3
import pandas as pd
import numpy as np
import os


def save_model(model, optimizer, metrics, path):
    """save model weight"""
    model_to_save = model.module if hasattr(model, 'module') else model
    save_dict = {
        'epoch': metrics.per_epoch['epoch'],
        'loss': metrics.per_epoch['loss'],
        'rec': metrics.per_epoch['rec'],
        'mrr': metrics.per_epoch['mrr'],
        'model': model_to_save.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(save_dict, path)


def load_model(model, device, path="./"):
    """load model weight"""
    model_to_load = model.module if hasattr(model, 'module') else model
    # load_dict = torch.load(path, map_location=lambda storage, loc: storage.cuda(device))
    # map_location changed to 'cpu' due to out of memory error
    # https://discuss.pytorch.org/t/out-of-memory-error-when-resume-training-even-though-my-gpu-is-empty/30757
    load_dict = torch.load(path, map_location='cpu')
    model_to_load.load_state_dict(load_dict['model'])
    if torch.cuda.is_available():
        model_to_load.cuda()
    return model_to_load


def save_jsonl(data: list, output_path: str, append=False):
    """Write list of objects to a JSON lines file."""
    mode = 'a+' if append else 'w'
    with open(output_path, mode, encoding='utf-8') as f:
        for line in data:
            json_record = json.dumps(line, ensure_ascii=False)
            f.write(json_record + '\n')
    print('Wrote {} records to {}'.format(len(data), output_path))


def load_jsonl(input_path: str) -> list:
    """Read list of objects from a JSON lines file."""
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.rstrip('\n|\r')))
    return data


def load_db(path: str, limit=None):
    """
    Return documents (column: text) and document ids (column: id)
    FEVER db returns wiki abstracts (single paragraphs) and document names (=id)
    CTK db returns paragraphs and paragraph ids
    """
    # Create the connection
    connection = sqlite3.connect(path)
    # Create the dataframe from a query
    if limit:
        data = pd.read_sql_query(f"SELECT * FROM documents LIMIT {limit}", connection)
    else:
        data = pd.read_sql_query("SELECT * FROM documents", connection)
    return list(data.text.values), list(data.id.values)


def load_json(path: str):
    with open(path, "r") as json_file:
        data = json.load(json_file)
    return data


def save_doc_chunks(path: str, doc_ids_chunks: list):
    """Save chunks as a pickle file."""
    with open(path, "ab") as handle:
        pickle.dump((
            doc_ids_chunks
        ), handle)


def load_doc_chunks(path: str):
    """Load prepared chunks as a pickle file."""
    ret = []
    if os.path.exists(path):
        with open(path, 'rb') as fr:
            try:
                while True:  # until EOF
                    ret.append(pickle.load(fr))
            except:
                return ret


def save_np_embeddings(data, path: str):
    with open(path, "wb") as handle:
        np.save(handle, data)


def load_np_embeddings(path: str):
    with open(path, "rb") as handle:
        return np.load(handle)