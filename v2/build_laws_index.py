import os
import re
import json
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

DATA_DIR = "./data"
INDEX_DIR = "./indices"
os.makedirs(INDEX_DIR, exist_ok=True)


def is_shitty_text(x):
    x = str(x).strip()
    if not x or len(x) < 5 or x in ["...", "(...)", "…"]: return True
    if re.fullmatch(r"[\W\d_]+", x): return True
    
    words = x.split()
    if len(words) == 1 and len(words[0]) < 4: return True
    if len(words) <= 2: return True 
    
    if "\ufffd" in x: return True 
    
    if len(x) > 40 and not re.search(r"[.!?;:]$", x):
        last = words[-1].lower()
        if last in {"und", "oder", "mit", "kein", "eine", "der", "die", "das"}: return True
        
    return False


def main():
    print("Loading laws_de...")
    df = pd.read_csv(os.path.join(DATA_DIR, "laws_de.csv"))
    
    df = df[~df["text"].apply(is_shitty_text)].copy()
    
    grouped = df.groupby("text")["citation"].apply(lambda x: list(set(x))).reset_index()
    
    grouped = grouped[grouped["citation"].apply(len) <= 10].reset_index(drop=True)
    
    mapping = {str(idx): {"text": row["text"], "citations": row["citation"]} 
               for idx, row in grouped.iterrows()}
    
    with open(os.path.join(INDEX_DIR, "laws_de_mapping.json"), "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    
    print(f"Embedding {len(grouped)} texts...")
    model = SentenceTransformer("BAAI/bge-m3")
    print("Embedding...")
    embeddings = model.encode(grouped["text"].tolist(), show_progress_bar=True, convert_to_numpy=True)
    
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim) 
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    
    faiss.write_index(index, os.path.join(INDEX_DIR, "laws_de.index"))
    print("laws_de index built successfully!")

if __name__ == "__main__":
    main()