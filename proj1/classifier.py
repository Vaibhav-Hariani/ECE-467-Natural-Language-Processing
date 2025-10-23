import sys
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import os
import random
from typing import Dict, List
import math
from dataclasses import dataclass


nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
# TOKENIZER = RegexpTokenizer(r'\w+')
STOPWORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()


@dataclass
class Document:
    path: str
    label: str
    tokens: List[str]
    tf_unnorm: Dict[str, int]
    num_terms: int

def load_docs(list_file_path, labeled=True):
    docs = []
    # determine base directory for relative paths in the list file
    base_dir = os.path.dirname(os.path.abspath(list_file_path))
    with open(list_file_path, 'r') as lf:
        for line in lf:
            line = line.strip()
            path, label = line.rsplit(' ', 1)
            if not labeled:
                label = None
            # keep the original (as-written) path for storage as a relative path
            original_path = path
            open_path = os.path.normpath(os.path.join(base_dir, original_path))

            with open(open_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
                doc = tokenize(text)
                doc.label = label
                doc.path = original_path
                docs.append(doc)
    return docs


def tokenize(doc):
    tf_dict = {}
    total_terms = 0
    for sent in nltk.sent_tokenize(doc):
        for tok in nltk.word_tokenize(sent):
            tok = tok.lower()
            tok = LEMMATIZER.lemmatize(tok)
            if tok not in STOPWORDS and tok.isalpha():
                tf_dict[tok] = tf_dict.get(tok, 0) + 1
                total_terms += 1
    tokens_list = list(tf_dict.keys())
    return Document(path=None, label=None, tokens=tokens_list, tf_unnorm=tf_dict, num_terms=total_terms)


def compute_idf(docs: List[Document]):
    freq_dict = {}
    N = len(docs)
    for doc in docs:
        terms = doc.tokens
        for t in terms:
            freq_dict[t] = freq_dict.get(t, 0) + 1   
    idf = {}
    doc_log = math.log(N)
    for term in freq_dict:
        idf[term] = doc_log - math.log(freq_dict[term])
    return idf


def compute_centroids(train_docs, idf):
    groups = {}
    for d in train_docs:
        if d.label in groups:
            groups[d.label].append(d)
        else:
            groups[d.label] = [d]

    centroids = {}
    for label, docs in groups.items():
        sum_vec: Dict[str, float] = {}
        for d in docs:
            for term, tfval in d.tf_unnorm.items():
                addition = (tfval/d.num_terms) * idf[term]
                sum_vec[term] = sum_vec.get(term, 0.0) + addition
        
        # average docs, compute norm
        n = len(docs)
        avg_vec: Dict[str, float] = {}
        sumsq = 0.0
        for t, val in sum_vec.items():
            a = val / n
            avg_vec[t] = a
            sumsq += a * a
        norm = math.sqrt(sumsq)
        centroid = {t: v / norm for t, v in avg_vec.items()}
        centroids[label] = centroid
    return centroids



## This is A.I Generated: Just a quick function for cross-validation splitting
def split_list(list_file_path, train_out, test_out, test_frac=0.2):
    rng = random.Random()

    # group lines by label
    groups = {}
    with open(list_file_path, 'r', encoding='utf-8') as lf:
        for line in lf:
            line = line.strip()
            if not line:
                continue
            # expect "path label"
            parts = line.rsplit(' ', 1)
            if len(parts) == 2:
                path, label = parts
            else:
                path = parts[0]
                label = ''
            groups.setdefault(label, []).append(line)

    train_lines = []
    test_lines = []
    for label, lines in groups.items():
        rng.shuffle(lines)
        n_test = int(len(lines) * test_frac)
        test_lines.extend(lines[:n_test])
        train_lines.extend(lines[n_test:])

    # ensure output directories exist
    train_dir = os.path.dirname(os.path.abspath(train_out))
    test_dir = os.path.dirname(os.path.abspath(test_out))
    if train_dir:
        os.makedirs(train_dir, exist_ok=True)
    if test_dir:
        os.makedirs(test_dir, exist_ok=True)

    with open(train_out, 'w', encoding='utf-8') as tf:
        for l in train_lines:
            tf.write(l + "\n")

    with open(test_out, 'w', encoding='utf-8') as tf:
        for l in test_lines:
            tf.write(l + "\n")

    return len(train_lines), len(test_lines)


## Modified to accept dict instead of vectors
def cosine_sim(a, b) -> float:
    s = 0.0
    for k, v in a.items():
        s += v * b.get(k, 0.0)
    return s


def classify(doc, idf, centroids):
    vec = {}
    for term, tfval in doc.tf_unnorm.items():
        if term in idf:
            vec[term] = (tfval / doc.num_terms) * idf[term]
    norm = math.sqrt(sum(v * v for v in vec.values()))
    vec = {t: v / norm for t, v in vec.items()}

    max_score = 0.0
    max_label = None
    for label, centroid in centroids.items():
        score = cosine_sim(vec, centroid)
        if score > max_score:
            max_score = score
            max_label = label
    return max_label, max_score


if __name__ == "__main__":
    train_list = 'proj1/corpus/corpus1_train.labels'
    test_list = 'proj1/corpus/corpus1_test.labels'
    if len(sys.argv) < 3:
        print("Usage: python classifier.py <train_list_file> <test_list_file> [<output_predictions_file>]")
    else:
        train_list = sys.argv[1]
        test_list = sys.argv[2]
    train_docs = load_docs(train_list, labeled=True)
    
    idf = compute_idf(train_docs)
    centroids = compute_centroids(train_docs, idf)

    test_docs = load_docs(test_list, labeled=False)    
    out_path = 'predictions.txt'
    out_path = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] else out_path

    with open(out_path, 'w', encoding='utf-8') as out:
        for d in test_docs:
            pred, score = classify(d, idf, centroids)
            out.write(f"{d.path} {pred}\n")
    print(f"Wrote predictions to {out_path}")

