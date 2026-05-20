import os
import re
import json
import pandas as pd
import faiss
import torch
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
    print("Loading court_considerations...")
    df = pd.read_csv(os.path.join(DATA_DIR, "court_considerations.csv"), low_memory=False)
    
    print(f"Original rows: {len(df)}")
    df = df[~df["text"].apply(is_shitty_text)].copy()
    
    print("Grouping by text...")
    grouped = df.groupby("text")["citation"].apply(lambda x: list(set(x))).reset_index()
    
    print("Dropping texts with >10 citations...")
    grouped = grouped[grouped["citation"].apply(len) <= 10].reset_index(drop=True)
    
    num_texts = len(grouped)
    print(f"Final unique valid texts to embed: {num_texts}")
    if num_texts == 0:
        raise ValueError("CRITICAL: 0 texts remaining after filtering!")
    
    mapping = {str(idx): {"text": row["text"], "citations": row["citation"]} 
               for idx, row in grouped.iterrows()}
    
    with open(os.path.join(INDEX_DIR, "courts_mapping.json"), "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    
    print("Loading embedding model (BAAI/bge-m3)...")
    
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    model = SentenceTransformer(
        "BAAI/bge-m3", 
        model_kwargs={"torch_dtype": torch.bfloat16}
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("Model initialized on 40GB H100 CUDA context with Native BF16.")
    
    print(f"Embedding {num_texts} texts...")
    
    embeddings = model.encode(
        grouped["text"].tolist(), 
        batch_size=128, 
        show_progress_bar=True, 
        convert_to_numpy=True,
        device=device
    )
    
    print("Building FAISS index structure...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim) # Inner product
    
    print("Executing L2 Normalization...")
    faiss.normalize_L2(embeddings)
    
    print("Adding vectors into index allocations...")
    index.add(embeddings)
    
    print("Saving FAISS index...")
    faiss.write_index(index, os.path.join(INDEX_DIR, "courts.index"))
    print("Success! courts index built cleanly.")

if __name__ == "__main__":
    main()