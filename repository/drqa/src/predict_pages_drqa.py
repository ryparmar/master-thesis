import argparse
import json
from multiprocessing.pool import ThreadPool

import os

from tqdm import tqdm

from drqa import retriever
# from prediction.retrieval.top_n import TopNDocsTopNSents
# from prediction.retrieval.fever_doc_db import FeverDocDB
from common.dataset.reader import JSONLineReader
# from rte.riedel.data import FEVERGoldFormatter, FEVERLabelSchema

import logging
logger = logging.getLogger(__name__)

# based on fever scripts/retrieval/ir.py
from drqa.retriever import DocDB, utils


# ----------------------------------------------------------------------------------------------------------------------
# https://github.com/UKPLab/conll2019-snopes-experiments/blob/102f4a05cfba781036bd3a7b06022246e53765ad/src/retrieval/fever_doc_db.py
# ----------------------------------------------------------------------------------------------------------------------
class FeverDocDB(DocDB):

    def __init__(self,path=None):
        super().__init__(path)

    def get_doc_lines(self, doc_id):
        """Fetch the raw text of the doc for 'doc_id'."""
        cursor = self.connection.cursor()
        cursor.execute(
            "SELECT lines FROM documents WHERE id = ?",
            (utils.normalize(doc_id),)
        )
        result = cursor.fetchone()
        cursor.close()
        return result if result is None else result[0]

    def get_non_empty_doc_ids(self):
        """Fetch all ids of docs stored in the db."""
        cursor = self.connection.cursor()
        cursor.execute("SELECT id FROM documents WHERE length(trim(text)) > 0")
        results = [r[0] for r in cursor.fetchall()]
        cursor.close()
        return results

# ----------------------------------------------------------------------------------------------------------------------
# RTE.RIEDEL.DATA
# ----------------------------------------------------------------------------------------------------------------------
import importlib.util
import os

from nltk import word_tokenize

from common.dataset.formatter import Formatter
from common.dataset.label_schema import LabelSchema
# from retrieval.filter_uninformative import uninformative


def preprocess(p):
    return p.replace(" ","_").replace("(","-LRB-").replace(")","-RRB-").replace(":","-COLON-").split("#")[0]

class FeverFormatter(Formatter):
    def __init__(self, index, label_schema, tokenizer=None,filtering=None):
        super().__init__(label_schema)
        self.index=index
        self.tokenize = tokenizer if tokenizer is not None else self.nltk_tokenizer
        self.filtering = None

        def import_module(filename):
            spec = importlib.util.spec_from_file_location('filter_doc', filename)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module

        if filtering is not None:
            if filtering:
                self.filtering = import_module(filtering).preprocess

    def nltk_tokenizer(self,text):
        return " ".join(word_tokenize(text))

class FEVERGoldFormatter(FeverFormatter):
    def format_line(self,line):
        annotation = None
        if "label" in line:
            annotation = line["label"]
        pages = []

        if 'predicted_sentences' in line:
            pages.extend([(ev[0], ev[1]) for ev in line["predicted_sentences"]])
        elif 'predicted_pages' in line:
            pages.extend([(ev[0],-1) for ev in line["predicted_pages"]])
        else:
            for evidence_group in line["evidence"]:
                pages.extend([(ev[2],ev[3]) for ev in evidence_group])

        if self.filtering is not None:
            for page,_ in pages:
                if self.filtering({"id":page}) is None:
                    return None
        if annotation is not None:
            return {"claim":self.tokenize(line["claim"]), "evidence": pages, "label":self.label_schema.get_id(annotation),"label_text":annotation}
        else:
            return {"claim":self.tokenize(line["claim"]), "evidence": pages, "label":None,"label_text":None}
 

class FEVERPredictionsFormatter(FeverFormatter):
    def format_line(self,line):
        annotation = line["label"]
        if annotation is None:
            annotation = line["verifiable"]

        pages = [preprocess(ev[0]) for ev in line["predicted_pages"]]
        return {"claim":self.tokenize(line["claim"]), "evidence": pages, "label":self.label_schema.get_id(annotation),"label_text":annotation}


class FEVERPredictions2Formatter(FeverFormatter):
    def format_line(self,line):
        annotation = line["label"]
        if annotation is None:
            annotation = line["verifiable"]

        if 'predicted_pages' in line:
            pages = [ev[0] for ev in line["predicted_pages"]]

        elif 'evidence' in line:
            pages = [ev[1] for ev in line["evidence"]]

        else:
            pages = []


        return {"claim":self.tokenize(line["claim"]), "evidence": pages, "label":self.label_schema.get_id(annotation),"label_text":annotation}


class FEVERLabelSchema(LabelSchema):
    def __init__(self):
        super().__init__(["supports","refutes","not enough info"])


# ----------------------------------------------------------------------------------------------------------------------
# ORIGINAL PREDICT_PAGES_DRQA
# ----------------------------------------------------------------------------------------------------------------------

class TopNDocs:
    def __init__(self, db, k, model):
        self.db = db
        self.k = k
        self.model = model
        self.ranker = retriever.get_class('tfidf')(tfidf_path=self.model)
        
    def retrieve_docs(self, claim):
        doc_names, doc_scores = self.ranker.closest_docs(claim, self.k)
        return doc_names, doc_scores

def process_line(method,line):
    pages, _ = method.retrieve_docs(line["claim"])
    line["predicted_pages"] = pages
    return line


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_map_function(parallel):
    return p.imap_unordered if parallel else map

if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    parser = argparse.ArgumentParser()
    parser.add_argument('--db', type=str, help='DRQA doc db file')
    parser.add_argument('--model', type=str, help='DRQA index file')
    parser.add_argument('--in-file', type=str, help='input dataset')
    parser.add_argument('--out-file', type=str, help='path to save output dataset')
    parser.add_argument('--max-page',type=int, help='number of pages to retrieve (k)')
    parser.add_argument('--parallel',type=str2bool,default=True)
    args = parser.parse_args()

    db = FeverDocDB(args.db)
    jlr = JSONLineReader()
    formatter = FEVERGoldFormatter(set(), FEVERLabelSchema())

    method = TopNDocs(db, args.max_page, args.model)

    processed = dict()

    with open(args.in_file,"r") as f, open(args.out_file, "w+", encoding="utf8") as out_file:
        lines = jlr.process(f)
        logger.info("processing lines")

        with ThreadPool() as p:
            for line in tqdm(get_map_function(args.parallel)(lambda line: process_line(method,line),lines), total=len(lines)):
                #out_file.write(json.dumps(line) + "\n")
                processed[line["id"]] = line

        logger.info("Done, writing to disk")

        for line in lines:
            out_file.write(json.dumps(processed[line["id"]], ensure_ascii=False) + "\n")