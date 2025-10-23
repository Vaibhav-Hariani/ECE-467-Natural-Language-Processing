
import sys
import nltk
import os
from typing import Dict, List
import math
from dataclasses import dataclass


@dataclass
class Document:
    path: str
    label: str
    tokens: List[str]
    tf_unnorm: Dict[str, int]
    num_terms: int

def load_docs(list_file_path, labeled=True):
    docs = []
    with open(list_file_path, 'r') as lf:
        for line in lf:
            line = line.strip()
            if not line:
                continue
            if labeled:
                path, label = line.rsplit(' ', 1)
            else:
                path = line
                label = None
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
                doc = tokenize(text)
                doc.label = label
                doc.path = path
                docs.append(doc)
    return docs


def tokenize(doc):
    tf_dict = {}
    total_terms = 0
    for sent in nltk.sent_tokenize(doc):
        for tok in nltk.word_tokenize(sent):
            tok = tok.lower()
            if tok.isalpha():
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
        # adding 1 to denominator for smoothing
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
    if norm > 0.0:
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
    if len(sys.argv) != 3:
        print("Usage: python classifier.py <train_list_file> <test_list_file>")
        sys.exit(1)
    train_list = sys.argv[1]
    test_list = sys.argv[2]
    train_docs = load_docs(train_list, labeled=True)
    if not train_docs:
        print(f"No training documents found in {train_list}")
        sys.exit(1)
    test_docs = load_docs(test_list, labeled=False)
    if not test_docs:
        print(f"No test documents found in {test_list}")
        sys.exit(1)

    idf = compute_idf(train_docs)
    centroids = compute_centroids(train_docs, idf)

    out_path = 'predictions.txt'
    if len(sys.argv) >= 4:
        out_path = sys.argv[3]
    with open(out_path, 'w', encoding='utf-8') as out:
        for d in test_docs:
            pred, score = classify(d, idf, centroids)
            out.write(f"{d.path} {pred}\n")
    print(f"Wrote predictions to {out_path}")

