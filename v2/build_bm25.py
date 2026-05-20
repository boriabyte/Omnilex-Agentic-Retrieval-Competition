import os
import json
import bm25s
import Stemmer
from pathlib import Path
from tqdm import tqdm

INDEX_DIR = Path("./indices")

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

def build_bm25():
    print("Loading JSON mappings...")
    with open(INDEX_DIR / "laws_de_mapping.json", "r", encoding="utf-8") as f:
        laws_map = json.load(f)
    with open(INDEX_DIR / "courts_mapping.json", "r", encoding="utf-8") as f:
        courts_map = json.load(f)

    corpus_texts = []
    doc_metadata = []

    print("Extracting texts from Laws...")
    for doc_id, data in tqdm(laws_map.items()):
        corpus_texts.append(data.get("text", ""))
        doc_metadata.append({"doc_id": doc_id, "source": "laws"})

    print("Extracting texts from Courts...")
    for doc_id, data in tqdm(courts_map.items()):
        corpus_texts.append(data.get("text", ""))
        doc_metadata.append({"doc_id": doc_id, "source": "courts"})

    print("Tokenizing corpus (this may take a few minutes)...")
    stemmer = Stemmer.Stemmer("german")
    
    # bm25s is heavily optimized for fast tokenization
    corpus_tokens = bm25s.tokenize(
        corpus_texts, 
        stopwords=list(GERMAN_STOPS), 
        stemmer=stemmer, 
        show_progress=True
    )

    print("Building BM25 index...")
    retriever = bm25s.BM25(corpus=corpus_tokens)
    retriever.index(corpus_tokens)

    print("Saving BM25 index and metadata...")
    bm25_path = INDEX_DIR / "bm25_index"
    retriever.save(str(bm25_path))
    
    with open(INDEX_DIR / "bm25_metadata.json", "w", encoding="utf-8") as f:
        json.dump(doc_metadata, f)
        
    print(f"BM25 index successfully saved to {bm25_path}!")

if __name__ == "__main__":
    build_bm25()