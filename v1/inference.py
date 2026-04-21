import pandas as pd
import numpy as np
import re
import json
import gc
import torch
import faiss
import bm25s
import Stemmer
from pathlib import Path
from collections import Counter
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import MarianMTModel, MarianTokenizer

INDEX_DIR = Path("./indexes")
DATA_DIR = Path("./data")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BM25_DEPTH = 500
DENSE_DEPTH = 300
MINE_TOP = 200    
HOP2_TOP = 30     
TARGET_CITES = 25 

_stemmer = Stemmer.Stemmer("german")
GERMAN_STOPS = set(
    "der die das den dem des ein eine einen einem einer und oder "
    "aber auch auf aus bei bis durch für gegen in mit nach ohne "
    "über um unter von vor zu zum zur als am an da dabei damit "
    "dann darauf darin dazu denn doch dort du er es hat hatte "
    "haben ich ihr ihre im ins ist ja jede jeder jedes kann kein "
    "keine können man mehr nicht noch nun nur ob schon sehr sich "
    "sie sind so solche sondern um und uns unser vom was weil "
    "welche wenn wer wie wird wir wo bereits diese diesem diesen "
    "dieser dieses".split()
)

CITE_PATTERNS = [
    re.compile(r"Art\.\s*\d+[a-z]?(?:\s+Abs\.\s*\d+)?(?:\s+(?:lit|Ziff)\.\s*[a-z])?\s+[A-ZÄÖÜ][A-ZÄÖÜa-zäöü]+"),
    re.compile(r"BGE\s+\d+\s+[IVX]+\s+\d+(?:\s+E\.\s*\d+(?:\.\d+)*)?"),
    re.compile(r"\d+[A-Z]_\d+/\d{4}(?:\s+E\.?\s*\d+(?:\.\d+)*)?"),
]


def calculate_metrics(pred, gold):
    p_set, g_set = set(pred), set(gold)
    if not p_set: return 0.0, 0.0, 0.0
    tp = len(p_set & g_set)
    precision = tp / len(p_set)
    recall = tp / len(g_set)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
    return precision, recall, f1

def mean_ap(preds, golds):
    aps = []
    for p, g in zip(preds, golds):
        g_set = set(g)
        hits, score = 0, 0.0
        for i, cite in enumerate(p):
            if cite in g_set:
                hits += 1
                score += hits / (i + 1)
        aps.append(score / len(g) if g else 0)
    return np.mean(aps)


class ModelProvider:
    def __init__(self):
        self._active = None
        self._model = None

    def _unload(self):
        if self._model is not None:
            del self._model
            torch.cuda.empty_cache()
            gc.collect()

    def translate(self, text: str) -> str:
        if self._active != "translator":
            self._unload()
            path = "Helsinki-NLP/opus-mt-en-de"
            tok = MarianTokenizer.from_pretrained(path)
            mod = MarianMTModel.from_pretrained(path).to(DEVICE)
            self._model = (tok, mod)
            self._active = "translator"

        tok, mod = self._model
        inp = tok(text, return_tensors="pt", truncation=True).to(DEVICE)

        with torch.no_grad():
            out = mod.generate(**inp)

        return tok.decode(out[0], skip_special_tokens=True)

    def encode(self, texts: list[str]) -> np.ndarray:
        if self._active != "encoder":
            self._unload()
            self._model = SentenceTransformer("BAAI/bge-m3", device=DEVICE)
            self._active = "encoder"
        return self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    def rerank(self, pairs: list[tuple[str, str]]) -> list[float]:
        if self._active != "reranker":
            self._unload()
            self._model = CrossEncoder("BAAI/bge-reranker-v2-m3", device=DEVICE)
            self._active = "reranker"
            
        return self._model.predict(pairs, show_progress_bar=False)

models = ModelProvider()

def extract_cites(text: str) -> list[str]:
    return [m.strip() for p in CITE_PATTERNS for m in p.findall(text)]

def get_doc_text(doc_id: int, laws_df, courts_df, n_laws) -> str:
    try:
        if doc_id < n_laws:
            return str(laws_df.iloc[doc_id]["text"])
        return str(courts_df.iloc[doc_id - n_laws]["text"])
    except: return ""

def hybrid_retrieve(query_en: str, query_de: str, bm25, faiss_idx):
    scores = {}
    tok = bm25s.tokenize([query_de], stemmer=_stemmer, stopwords=list(GERMAN_STOPS))

    res, _ = bm25.retrieve(tok, k=BM25_DEPTH)

    for rank, did in enumerate(res[0]):
        scores[int(did)] = scores.get(int(did), 0) + 1.0 / (60 + rank)

    embs = models.encode([query_en, query_de])

    for emb in embs:

        _, ids = faiss_idx.search(emb.reshape(1, -1).astype("float32"), DENSE_DEPTH)
        for rank, did in enumerate(ids[0]):
            if did != -1:
                scores[int(did)] = scores.get(int(did), 0) + 1.0 / (60 + rank)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

def mine_citations(ranked_docs, laws_df, courts_df, n_laws, clookup):
    h1 = Counter()

    for did, _ in ranked_docs[:MINE_TOP]:

        text = get_doc_text(did, laws_df, courts_df, n_laws)
        for cite in extract_cites(text):
            if cite in clookup: 
                h1[cite] += 1

    h2 = Counter()
    top_court_refs = [c for c, _ in h1.most_common(100) if "BGE" in c or "_" in c]

    for cite in top_court_refs[:HOP2_TOP]:
        did = clookup[cite][0]
        text = get_doc_text(did, laws_df, courts_df, n_laws)
        for c in extract_cites(text):
            if c in clookup: h2[c] += 1

    return h1, h2

def main():
    with open(INDEX_DIR / "corpus_offsets.json") as f:
        n_laws = json.load(f)["n_laws"]

    laws_df = pd.read_parquet(INDEX_DIR / "laws_corpus.parquet")
    courts_df = pd.read_parquet(INDEX_DIR / "courts_corpus.parquet")

    with open(INDEX_DIR / "citation_lookup.json") as f:
        clookup = json.load(f)

    bm25 = bm25s.BM25.load(str(INDEX_DIR / "full_bm25"))
    faiss_idx = faiss.read_index(str(INDEX_DIR / "full_corpus_bge_m3_final.index"))

    query_file = "val.csv"

    df = pd.read_csv(DATA_DIR / query_file)
    has_gold = "gold_citations" in df.columns
    
    all_preds, all_golds, submission_data = [], [], []

    for _, row in df.iterrows():
        qid, query = str(row["query_id"]), str(row["query"])

        # translated to german
        query_de = models.translate(query)
        
        ranked = hybrid_retrieve(query, query_de, bm25, faiss_idx)

        h1, h2 = mine_citations(ranked, laws_df, courts_df, n_laws, clookup)
        
        c_scores = {c: f * 10 for c, f in h1.items()}

        for c, f in h2.items():
            c_scores[c] = c_scores.get(c, 0) + f * 3

        for c in set(extract_cites(query) + extract_cites(query_de)):
            if c in clookup:
                c_scores[c] = c_scores.get(c, 0) + 500
            
        candidates = sorted(c_scores.keys(), key=lambda x: c_scores[x], reverse=True)[:50]

        if candidates:
            pairs = [(query, f"{c} {get_doc_text(clookup[c][0], laws_df, courts_df, n_laws)[:400]}") for c in candidates]
            rr_scores = models.rerank(pairs)
            final = [c for c, _ in sorted(zip(candidates, rr_scores), key=lambda x: x[1], reverse=True)][:TARGET_CITES]

        else: 
            final = []
            
        submission_data.append({"query_id": qid, "predicted_citations": ";".join(final)})
        
        if has_gold:
            gold = [c.strip() for c in str(row["gold_citations"]).split(";") if c.strip()]
            all_preds.append(final)
            all_golds.append(gold)
            p, r, f1 = calculate_metrics(final, gold)
            print(f"[{qid}] F1: {f1:.3f} | Prec: {p:.3f} | Rec: {r:.3f}")

    pd.DataFrame(submission_data).to_csv("submission.csv", index=False)

if __name__ == "__main__":
    main()