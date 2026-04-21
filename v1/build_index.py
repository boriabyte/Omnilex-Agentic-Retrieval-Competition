import pandas as pd
import numpy as np
import json
import gc
import torch
import faiss
import bm25s
from pathlib import Path
import Stemmer
from sentence_transformers import SentenceTransformer

LAWS_CSV   = Path("./data/laws_de.csv")
COURTS_CSV = Path("./data/court_considerations.csv")
INDEX_DIR  = Path("./indexes")
INDEX_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

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


def load_corpus():
    print("Loading laws...")
    laws = pd.read_csv(LAWS_CSV)
    laws["text"]  = laws["text"].fillna("").astype(str)
    laws["citation"] = laws["citation"].fillna("").astype(str).str.strip()
    laws["title"] = laws["title"].fillna("").astype(str) if "title" in laws.columns else ""
    laws["source_type"] = "law"
    print(f"  {len(laws):,} law articles")

    print("Loading court considerations...")
    courts = pd.read_csv(COURTS_CSV)
    courts["text"]  = courts["text"].fillna("").astype(str)
    courts["citation"] = courts["citation"].fillna("").astype(str).str.strip()
    courts["title"] = ""
    courts["source_type"] = "court"
    print(f"  {len(courts):,} court considerations")

    full = pd.concat([laws, courts], ignore_index=True)
    full["doc_id"] = range(len(full))

    n_laws   = len(laws)
    n_courts = len(courts)
    print(f"  Full corpus: {len(full):,} (laws 0..{n_laws-1}, courts {n_laws}..{n_laws+n_courts-1})")

    return full, n_laws, n_courts



def build_citation_lookup(corpus: pd.DataFrame) -> dict:
    print("Building lookup table (vectorized)...")
    lookup = corpus.groupby("citation")["doc_id"].apply(list).to_dict()
    return lookup


def build_bm25(corpus: pd.DataFrame):
    print(f"Building BM25 index on full corpus ({len(corpus):,} docs)...")
    stemmer = Stemmer.Stemmer("german") 

    texts = (corpus["citation"] + " " + corpus["text"]).tolist()
    tokenized = bm25s.tokenize(texts, stopwords=list(GERMAN_STOPS), stemmer=stemmer)

    retriever = bm25s.BM25(method="lucene")
    retriever.index(tokenized)

    save_path = str(INDEX_DIR / "full_bm25_w_lucene")
    retriever.save(save_path)
    return retriever


EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM   = 1024
BATCH_SIZE      = 256

def build_faiss(corpus: pd.DataFrame):
    try:
        sub_corpus = corpus
        print(f"Building FAISS index (Dense)...", flush=True)

        model = SentenceTransformer(EMBEDDING_MODEL, device=DEVICE)
        
        model.max_seq_length = 512 
        
        print("  Pre-truncating text...", flush=True)
        texts = (sub_corpus["citation"] + " " + sub_corpus["text"]).str[:2000].tolist()

        print(f"  Encoding {len(texts):,} rows...", flush=True)
        with torch.no_grad():
            embeddings = model.encode(
                texts,
                batch_size=128,      
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True
            )

        index = faiss.IndexFlatIP(EMBEDDING_DIM)
        index.add(embeddings.astype("float32"))
        
        faiss.write_index(index, str(INDEX_DIR / "full_corpus_bge_m3_final.index"))
        del model, embeddings
        torch.cuda.empty_cache()
        gc.collect()    
        return index

    except Exception as e:
        faiss.write_index(index, str(INDEX_DIR / "full_corpus_bge_m3_final_partial.index"))
        print(f"Error during FAISS indexing: {e}")


def save_metadata(corpus, n_laws, n_courts, citation_lookup):
    meta = corpus[["doc_id", "citation", "source_type"]].copy()
    meta.to_parquet(INDEX_DIR / "full_corpus_meta.parquet", index=False)

    laws = corpus.iloc[:n_laws].copy()
    laws.to_parquet(INDEX_DIR / "laws_corpus.parquet", index=False)

    courts = corpus.iloc[n_laws:][["doc_id", "citation", "text"]].copy()
    courts.to_parquet(INDEX_DIR / "courts_corpus.parquet", index=False,
                      compression="zstd")

    with open(INDEX_DIR / "citation_lookup.json", "w", encoding="utf-8") as f:
        json.dump(citation_lookup, f, ensure_ascii=False)

    offsets = {"n_laws": n_laws, "n_courts": n_courts, "n_total": n_laws + n_courts}
    with open(INDEX_DIR / "corpus_offsets.json", "w") as f:
        json.dump(offsets, f)

    print(f"  Saved metadata: laws_corpus.parquet, courts_corpus.parquet, "
          f"full_corpus_meta.parquet, citation_lookup.json, corpus_offsets.json")



if __name__ == "__main__":
    corpus, n_laws, n_courts = load_corpus()
    citation_lookup = build_citation_lookup(corpus)

    bm25_retriever = build_bm25(corpus)   
    # faiss_index    = build_faiss(corpus)
    # laws_faiss_index = build_faiss(corpus.iloc[:n_laws])

    # save_metadata(corpus, n_laws, n_courts, citation_lookup)