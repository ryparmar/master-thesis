from collections import Counter, OrderedDict
import pandas as pd

from nltk.util import skipgrams
import math
import json
import os


def process_csv(claims: str, labels: str):
    # Read csv files
    _claims = pd.read_csv(claims)
    _claims.dropna(subset=['id', 'claim'], inplace=True)
    _labels = pd.read_csv(labels)

    # Basic data processing
    _labels.id = _labels.claim
    df = _claims.merge(_labels[['id', 'label']], on='id', how='left')
    df = df.dropna(subset=['claim', 'label'])
    df = df[['id', 'claim', 'label']]
    df.reset_index(inplace=True, drop=True)
    return df


def read_jsonl_files(files: list) -> dict:
    claims, labels = [], []
    for file in files:
        with open(file) as fr:
            for line in fr:
                d = json.loads(line)
                claims.append(d['claim'])
                labels.append(d['label'])
    assert len(claims) == len(labels)
    # return claims, labels
    d = {'claim': claims, 'label': labels}
    return d


def process_jsonl_in_folder(path: str, split: str):
    """Read all jsonl files in given path and process them into a single DataFrame."""
    if split == 'all':
        files = [os.path.join(path, file) for file in os.listdir(path) if file.endswith(".jsonl")]
    else:
        files = [os.path.join(path, split, '.jsonl')]
    print(f"Reading from: {files}.")
    d = read_jsonl_files(files)
    df = pd.DataFrame(d)
    return df


def get_k_fold(df, k=10):
    """
    This approach assumes a balanced dataset with regard to the frequency of each label. If executed on
    an imbalanced dataset, a given cue’s productivity would be dominated by the most frequent label, 
    not because it is actually more likely to appear in a claim with that label but solely because of the label
    is more frequent in overall.

    V clanku undersampluji majority classes, a to opakuji 10x, aby dostali robustnejsi odhad. 
    """
    SAMPLE_SIZE = min(df['label'].value_counts())
    SAMPLES = []
    for _ in range(k):
        df_to_join = []
        for label in df.label.unique():
            df_to_join.append(df[df.label == label].sample(SAMPLE_SIZE)[['claim', 'label']])
        
        SAMPLES.append(pd.concat(df_to_join).reset_index(drop=True))
    return SAMPLES


def create_unigrams(data):
    return [[ii.strip('.') for ii in claim.split()] for claim in data['claim'] if isinstance(claim, str)]


def create_bigrams(data):
    return [[i.strip('.') + ' ' + ii.strip('.')
            for i, ii in zip(claim.split()[:-1], claim.split()[1:])] 
            for claim in data['claim'] if isinstance(claim, str)]


def create_wordpieces(data, tokenizer):
    return [tokenizer.tokenize(claim.rstrip('.')) for claim in data['claim'] if isinstance(claim, str)]


def get_cues(data, cue, cv, tokenizer):
    """
    Cue = bias v datech, ktery ulehcuje ML rozhodovani jen na zaklade nej. (Napriklad slovo 'not' v anglickych
    datech bude pritono v claimech s labelem takoveho rozdeleni: 80% REFUTES, 5% SUPPORTS, 15% NOT ENOUGH INFO - 
    coz vypada na bias vuci REFUTES class; ML predikuje REFUTES vzdy v pripade pritomnosti 'not', coz neni zadouci!)
    
    Valid cues: unigram, bigram, wordpieces
    """
    if cue == 'unigram':
        if not cv:
            unigrams = create_unigrams(data)
        else:
            unigrams = [create_unigrams(sample) for sample in data]
        return unigrams        
    elif cue == 'bigram':
        if not cv:
            bigrams = create_bigrams(data)
        else:
            bigrams = [create_bigrams(sample) for sample in data]
        return bigrams
    elif cue == 'wordpiece':
        if not cv:
            wordpieces = create_wordpieces(data, tokenizer)
        else:
            wordpieces = [create_wordpieces(sample, tokenizer) for sample in data]
        return wordpieces


def get_applicability(cues, cv):
    """
    Cue Applicability = the absolute number of claims in the dataset that contain the cue irrespective of their label 
                        = v kolika claimech je cue pritomna
    """
    def calculate_applicability(cues):
        tmp = []
        for ii in [set(i) for i in cues]:
            tmp += ii
        return Counter(tmp)

    if not cv:
        applicability = calculate_applicability(cues)
    else:
        applicability = [calculate_applicability(cues_in_fold) for cues_in_fold in cues]
    return applicability


def get_productivity(data, cues, applicability, cv):
    """
    Cue Productivity = is the frequency of the most common label across the claims that contain the cue 
                        = cetnost nejcastejsiho labelu pro cue
    """
    def get_max(values: dict) -> (str, int):
        label, max_count = None, 0
        for k, v in values.items():
            if v > max_count:
                max_count = v
                label = k
        return label, max_count

    def calculate_productivity(data, cues, applicability):
        counts_per_cue = {}  # rozdeleni labelu pro jednotlive cues
        for i, words in enumerate([set(i) for i in cues]):
            for w in words:
                if w not in counts_per_cue:
                    counts_per_cue[w] = {}
                if data['label'][i] not in counts_per_cue[w]:
                    counts_per_cue[w][data['label'][i]] = 1
                else:
                    counts_per_cue[w][data['label'][i]] += 1

        max_counts = {k: get_max(v) for k, v in counts_per_cue.items()}
        productivity = {k: v[1] / applicability[k] for k, v in max_counts.items()} 
        return OrderedDict(sorted(productivity.items(), key=lambda kv: kv[1], reverse=True))

    if not cv:
        productivity = calculate_productivity(data, cues, applicability)
    else:
        productivity = [calculate_productivity(data[i], cues[i], applicability[i]) for i in range(len(data))]
    return productivity


def get_coverage(data, applicability, cv):
    """
    Cue Coverage = applicability of a cue / total number of claims
                    = v kolika claimech je cue pritomna / pocet claimu
    """
    def calculate_coverage(data, applicability):
        return {k: v / len(data) for k, v in applicability.items()}

    if not cv:
        coverage = calculate_coverage(data, applicability)
        coverage = OrderedDict(sorted(coverage.items(), key=lambda kv: kv[1], reverse=True))
    else:
        coverage = [calculate_coverage(data[i], applicability[i]) for i in range(len(data))]
        # sorted_cov = [OrderedDict(sorted(coverage[i].items(), key=lambda kv: kv[1], reverse=True)) 
        #                 for i in range(len(data))]  # je toto k necemu? 
    return coverage


def get_utility(productivity, cv, num_labels):
    """
    Cue Utility =(for ML algorithm: the higher utility the easier decision for ML alg)

    In order to compare above metrics between datasets utility is the metric to go. A cue is only useful to 
    a machine learning model if productivity_k > 1 / m, where m is the number of possible labels 
    (supports, refutes, not enough info in FEVER dataset).
    """
    if not cv:
        utility = {k: v - 1/num_labels for k, v in productivity.items()}
    else:
        utility = [{k: v - 1/num_labels for k, v in productivity[i].items()} for i in range(len(productivity))]
    return utility


def get_result_frame(data, cue_form, cv, num_unique_labels, tokenizer):
    """
    Prepare dataframe with computed metrics.
    """
    def create_result_frame(productivity, utility, coverage):
        res = pd.DataFrame.from_dict(productivity, orient='index', columns=['productivity']).join(
                [pd.DataFrame.from_dict(utility, orient='index', columns=['utility']),
                pd.DataFrame.from_dict(coverage, orient='index', columns=['coverage'])])

        res['harmonic_mean'] = res.apply(lambda x: 2 / (1/x['productivity'] + 1/x['coverage']), axis=1)
        return res.sort_values('harmonic_mean', ascending=False)

    # Calculate the metrics
    cues = get_cues(data, cue_form.lower().strip(), cv, tokenizer)
    applicability = get_applicability(cues, cv)
    productivity = get_productivity(data, cues, applicability, cv)
    coverage = get_coverage(data, applicability, cv)
    # print(cues[0][:2], '\n', applicability[0]['v'], '\n', productivity[0]['v'], '\n', coverage[0]['v'])
    utility = get_utility(productivity, cv, num_unique_labels)

    if not cv:
        res = create_result_frame(productivity, utility, coverage)
    else:
        res_folds = [create_result_frame(productivity[i], utility[i], coverage[i]) for i in range(len(data))]
        # Merge all the samples and compute the estimate
        res = res_folds[0]
        for i in range(1, len(data)):
            res = res.add(res_folds[i], fill_value=0)
        res = res.div(len(data))
        res = res.sort_values('harmonic_mean', ascending=False)
    return res.round(4)  # return rounded dataframe with results


def claim_to_unigrams(claim):
    return [ii.strip('.') for ii in claim.split()] if isinstance(claim, str) else None

def claim_to_wordpieces(claim, tokenizer):
    return tokenizer.tokenize(claim.rstrip('.')) if isinstance(claim, str) else None

def create_skipgrams(data, cue: str, skip: int, tokenizer=None):
    """
    Cue is represented by skipgram.
    Creates skipgrams and cue counts per label and per document.

    cue = Cue representation: bigram, trigram
    skip = number of skipped tokens
    Note: if skip == 4, then skipgrams function generates all the skipgrams with 0, 1, 2, 3 and 4 skipped tokens.
    
    Returns:
    skipgrams_per_label = {'V Hradci': {'Supports': 4, 'Refutes': 1}, ...}
    skipgrams_total = {'V Hradci': 5, ...}
    skipgrams_document_frequency = {'V Hradci': 2, ...}
    total_documents = total number of documents
    """
    skipgrams_per_label, skipgrams_total = {}, {}
    skipgrams_document_frequency, total_documents = {}, len(data['claim'])
    rep2int = {'unigram': 1, 'wordpiece': 1, 'bigram': 2, 'trigram': 3}
    for i, claim in enumerate(data['claim']):
        # TODO rewrite -- added expost and slightly dumb? (calculating same thing as in the applicability?!) 
        _skipgrams = (skipgrams(claim.split(), rep2int[cue], skip) if rep2int[cue] > 1 else 
                    claim_to_unigrams(claim) if cue == 'unigram' else claim_to_wordpieces(claim, tokenizer))
        for skipgram in _skipgrams:
            skipgram = " ".join(list(skipgram)) if rep2int[cue] > 1 else "".join(list(skipgram))
            # Count skipgrams per cue
            if skipgram in skipgrams_total:
                skipgrams_total[skipgram] += 1
            else:
                skipgrams_total[skipgram] = 1
            if skipgram in skipgrams_per_label:
                if data['label'][i] in skipgrams_per_label[skipgram]:
                    skipgrams_per_label[skipgram][data['label'][i]] += 1
                else:
                    skipgrams_per_label[skipgram][data['label'][i]] = 1
            else:
                skipgrams_per_label[skipgram] = {data['label'][i]: 1}
            
            # Count document frequency per cue
            if skipgram in skipgrams_document_frequency:
                skipgrams_document_frequency[skipgram].add(i)
            else:
                skipgrams_document_frequency[skipgram] = set([i])
                    
    # Count the distinct docs         
    for k, v in skipgrams_document_frequency.items():
        skipgrams_document_frequency[k] = len(v)

    return skipgrams_per_label, skipgrams_total, skipgrams_document_frequency, total_documents


def compute_normalised_dist(nominator: dict, denominator: dict):
    """Returns normalised distribution over cues and labels"""
    return {cue: 
            {label: count / total for label, count in nominator[cue].items()} 
            for cue, total in denominator.items()}


def entropy(x: dict):
    return sum([v * math.log(v, 10) for k, v in x.items()])


def lambda_h(N: dict):
    """Information based factor (entropy)"""
    h = {k: 1 + entropy(v) for k, v in N.items()}
    return h


def lambda_f(s: int, doc_freq_per_cue: dict, total_docs: int):
    """
    Frequency-based scaling factor
    equivalent to normalized/scaled document frequency of a cue 
    = the number of documents in which is the cue present
    """
    f = {k: math.pow((v / total_docs), (1/s)) for k, v in doc_freq_per_cue.items()}
    return f


def DCI(lamh: dict, lamf: dict):
    """Dataset-weighted Cue Information"""
    dci = {k: math.sqrt(vh * lamf[k]) for k, vh in lamh.items()}
    return dci

def calculate_dci(data, cue='bigram', skip=4, hyperpar_s=3, tokenizer=None):
    """Dataset-weighted Cue Information"""
    skipgrams_label, skipgrams_total, skipgrams_df, skipgrams_total_docs = create_skipgrams(data, cue, 4, tokenizer)
    N = compute_normalised_dist(skipgrams_label, skipgrams_total)
    lambh = lambda_h(N)
    lambf = lambda_f(hyperpar_s, skipgrams_df, skipgrams_total_docs)
    dci = DCI(lambh, lambf)
    dci = sorted(dci.items(), key=lambda kv: kv[1], reverse=True)  # dict -> list of tuples
    dci_df = pd.DataFrame(dci, columns=['Cue', 'DCI'])
    return dci_df.round(4)
