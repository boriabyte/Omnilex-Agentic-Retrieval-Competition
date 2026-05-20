import os
import re
import json
import gc
import torch
import faiss
import bm25s
import Stemmer
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoModelForCausalLM, AutoTokenizer


INDEX_DIR = Path("./indices")
DATA_DIR = Path("./data")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BM25_DEPTH = 300
DENSE_DEPTH = 300
MINE_TOP = 200    
HOP2_TOP = 30     
TARGET_CITES = 35  

CITE_PATTERNS = [
    re.compile(r"Art\.\s*\d+[a-z]?(?:\s+Abs\.\s*\d+)?(?:\s+(?:lit|Ziff)\.\s*[a-z])?\s+[A-ZÄÖÜ][A-ZÄÖÜa-zäöü]+"),
    re.compile(r"BGE\s+\d+\s+[IVX]+\s+\d+(?:\s+E\.\s*\d+(?:\.\d+)*)?"),
    re.compile(r"\d+[A-Z]_\d+/\d{4}(?:\s+E\.?\s*\d+(?:\.\d+)*)?"),
]

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
_stemmer = Stemmer.Stemmer("german")

def free_memory():
    torch.cuda.empty_cache()
    gc.collect()


def calculate_metrics(pred, gold):
    p_set, g_set = set(pred), set(gold)
    if not p_set: return 0.0, 0.0, 0.0
    tp = len(p_set & g_set)
    precision = tp / len(p_set)
    recall = tp / len(g_set) if g_set else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
    return precision, recall, f1


LAW_TRANSLATIONS = {
    "LAI": "IVG"
}

def extract_cites(text: str) -> list[str]:
    raw_cites = [m.strip() for p in CITE_PATTERNS for m in p.findall(str(text))]
    expanded_cites = set(raw_cites)
    
    for cite in raw_cites:
        normalized_cite = cite
        for fr_it, de in LAW_TRANSLATIONS.items():
            if normalized_cite.upper().endswith(fr_it):
                normalized_cite = re.sub(rf"\b{fr_it}\b", de, normalized_cite, flags=re.IGNORECASE)
                expanded_cites.add(normalized_cite)
                break
                
        for base in [cite, normalized_cite]:
            no_lit = re.sub(r"\s+(?:lit|Ziff)\.\s*[a-z0-9]+\b", "", base)
            if no_lit != base:
                expanded_cites.add(no_lit)
                
            no_abs = re.sub(r"\s+Abs\.\s*\d+", "", no_lit)
            if no_abs != no_lit:
                expanded_cites.add(no_abs)
                
    return list(expanded_cites)


def run_qwen_translation(queries):
    print("\n--- PHASE 1: Qwen Legal Translation & Rephrasing ---")
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto"
    )
    
    system_prompt = (
        "You are an expert translator specializing in Swiss Law. Your task is to translate the following "
        "English legal scenario into formal, precise Swiss German legal text. "
        "CRITICAL: If the English text mentions any specific laws, articles, or court cases, format them "
        "according to exact Swiss citation rules (e.g., 'Art. 221 Abs. 1 lit. b StPO', 'BGE 137 IV 122'). "
        "Output ONLY the translated German text. Do not include any conversational openings or explanations."
    )
    
    translated_queries = []
    for query in tqdm(queries, desc="Qwen Translating"):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([text], return_tensors="pt").to(DEVICE)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.1, do_sample=False)
            
        new_tokens = outputs[0][len(inputs.input_ids[0]):]
        translated_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        translated_queries.append(translated_text)
        
    del model
    del tokenizer
    free_memory()
    return translated_queries


def main():
    query_file = "test.csv"
    print(f"Reading queries from {DATA_DIR / query_file}...")
    df = pd.read_csv(DATA_DIR / query_file)
    has_gold = "gold_citations" in df.columns

    # 1. Run Qwen Translation
    df['query_de'] = run_qwen_translation(df['query'].tolist())

    print("\n--- PHASE 2: FAISS, BM25 & Graph Mining ---")
    print("Loading FAISS indices and JSON mappings...")
    laws_idx = faiss.read_index(str(INDEX_DIR / "laws_de.index"))
    courts_idx = faiss.read_index(str(INDEX_DIR / "courts.index"))

    with open(INDEX_DIR / "laws_de_mapping.json", "r", encoding="utf-8") as f:
        laws_map = json.load(f)
    with open(INDEX_DIR / "courts_mapping.json", "r", encoding="utf-8") as f:
        courts_map = json.load(f)

    print(f"[LOG] Total items in laws mapping JSON:   {len(laws_map)}")
    print(f"[LOG] Total items in courts mapping JSON: {len(courts_map)}")

    print("Loading BM25 Index...")
    bm25_retriever = bm25s.BM25.load(str(INDEX_DIR / "bm25_index"), load_corpus=False)
    with open(INDEX_DIR / "bm25_metadata.json", "r", encoding="utf-8") as f:
        bm25_metadata = json.load(f)

    clookup = {}
    def add_to_lookup(mapping, source_name):
        for doc_id, data in mapping.items():
            for cit in data.get("citations", []):
                if cit not in clookup: 
                    clookup[cit] = []
                    
                clookup[cit].append((doc_id, source_name))

    add_to_lookup(laws_map, "laws")
    add_to_lookup(courts_map, "courts")

    def get_doc_text(doc_id, source):
        if source == "laws": 
            return laws_map.get(str(doc_id), {}).get("text", "")
        
        return courts_map.get(str(doc_id), {}).get("text", "")

    print("Loading BGE-M3 Dense Embedder...")
    embedder = SentenceTransformer("BAAI/bge-m3", device=DEVICE)
    
    all_candidates_lists = []
    all_source_attributions = []  

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Mining Candidates"):
        query_en = str(row["query"])
        query_de = str(row["query_de"])
        
        scores = {}
        bm25_doc_keys = set()
        dense_doc_keys = set()
        
        # bm25
        tokens = bm25s.tokenize([query_de], stopwords=list(GERMAN_STOPS), stemmer=_stemmer)
        bm25_res, _ = bm25_retriever.retrieve(tokens, k=BM25_DEPTH)
        
        for rank, bm25_idx in enumerate(bm25_res[0]):
            meta = bm25_metadata[int(bm25_idx)]
            key = (str(meta["doc_id"]), meta["source"])
            bm25_doc_keys.add(key)
            scores[key] = scores.get(key, 0) + 1.0 / (60 + rank) 

        # dense inex
        embs = embedder.encode([query_en, query_de], normalize_embeddings=True, show_progress_bar=False)
        for emb in embs:
            emb_reshaped = emb.reshape(1, -1).astype("float32")
            
            _, l_ids = laws_idx.search(emb_reshaped, DENSE_DEPTH)
            for rank, did in enumerate(l_ids[0]):
                if did != -1:
                    key = (str(did), "laws")
                    dense_doc_keys.add(key)
                    scores[key] = scores.get(key, 0) + 1.0 / (60 + rank) # RRF Fusion
                    
            _, c_ids = courts_idx.search(emb_reshaped, DENSE_DEPTH)
            for rank, did in enumerate(c_ids[0]):
                if did != -1:
                    key = (str(did), "courts")
                    dense_doc_keys.add(key)
                    scores[key] = scores.get(key, 0) + 1.0 / (60 + rank) # RRF Fusion

        citation_sources = {} 
        
        explicit_query_cites = set(extract_cites(query_en) + extract_cites(query_de))
        for c in explicit_query_cites:
            if c in clookup:
                citation_sources.setdefault(c, set()).add("regex")

        ranked_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # graph mining - hop 1
        h1 = Counter()
        for (did, source), _ in ranked_docs[:MINE_TOP]:
            doc_key = (str(did), source)
            text = get_doc_text(did, source)
            for cite in extract_cites(text):
                if cite in clookup: 
                    h1[cite] += 1

                    if doc_key in bm25_doc_keys:
                        citation_sources.setdefault(cite, set()).add("bm25")

                    if doc_key in dense_doc_keys:
                        citation_sources.setdefault(cite, set()).add("dense")

        # hop 2
        h2 = Counter()
        top_court_refs = [c for c, _ in h1.most_common(100) if "BGE" in c or "_" in c]
        for cite in top_court_refs[:HOP2_TOP]:
            if cite in clookup:
                did, source = clookup[cite][0]
                text = get_doc_text(did, source)
                for c in extract_cites(text):
                    if c in clookup: 
                        h2[c] += 1
                        parent_srcs = citation_sources.get(cite, set())
                        for p_src in parent_srcs:
                            if p_src in ["bm25", "dense"]:
                                citation_sources.setdefault(c, set()).add(p_src)

        # freq based score for hop 1
        c_scores = {c: f * 10 for c, f in h1.items()}
        for c, f in h2.items():
            c_scores[c] = c_scores.get(c, 0) + f * 3

        # regex exact match
        for c in explicit_query_cites:
            if c in clookup:
                # extreme boost for regex matches in the original query
                c_scores[c] = c_scores.get(c, 0) + 500 
                
                # recursion depth 1
                for did, source in clookup[c]:
                    text = get_doc_text(did, source)
                    for expanded_cite in extract_cites(text):

                        if expanded_cite in clookup and expanded_cite not in explicit_query_cites:
                            # recursion depth 2
                            c_scores[expanded_cite] = c_scores.get(expanded_cite, 0) + 200
                            citation_sources.setdefault(expanded_cite, set()).add("regex")

        candidates = sorted(c_scores.keys(), key=lambda x: c_scores[x], reverse=True)[:50]
        all_candidates_lists.append(candidates)
        all_source_attributions.append(citation_sources)

    del embedder
    free_memory()

    print("\n--- PHASE 3: Cross-Encoder Reranking (Dynamic Thresholding) ---")
    reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", device=DEVICE)
    
    all_preds, all_golds, submission_data = [], [], []
    
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Reranking Candidates"):
        qid = str(row["query_id"])
        query_en = str(row["query"])
        query_de = str(row["query_de"])
        candidates = all_candidates_lists[i]
        citation_sources = all_source_attributions[i]
        
        if candidates:
            pairs = []
            for c in candidates:
                did, source = clookup[c][0]
                text = get_doc_text(did, source)
                pairs.append((query_en, f"{c} {text[:400]}"))
                
            rr_scores = reranker.predict(pairs, show_progress_bar=False)
            sorted_pairs = sorted(zip(candidates, rr_scores), key=lambda x: x[1], reverse=True)
            
            final = []
            for rank_idx, (c, score) in enumerate(sorted_pairs):
                if rank_idx < 3:
                    final.append(c)
                elif score > 0.0 and len(final) < TARGET_CITES:
                    final.append(c)
        else: 
            final = []
            
        submission_data.append({"query_id": qid, "predicted_citations": ";".join(final)})
        
        log_lines = [
            f"\n" + "="*50,
            f"QUERY ID: {qid}",
            f"  [ORIGINAL EN]:  {query_en[:120]}...",
            f"  [REPHRASED DE]: {query_de[:120]}..."
        ]

        if has_gold:
            gold = [c.strip() for c in str(row["gold_citations"]).split(";") if c.strip()]
            all_preds.append(final)
            all_golds.append(gold)

            gold_set = set(gold)
            hits_in_candidates = len(gold_set & set(candidates))
            hits_in_final = len(gold_set & set(final))
            
            from_regex = sum(1 for gc in gold if "regex" in citation_sources.get(gc, set()))
            from_bm25  = sum(1 for gc in gold if "bm25" in citation_sources.get(gc, set()))
            from_dense = sum(1 for gc in gold if "dense" in citation_sources.get(gc, set()))

            prec, rec, f1 = calculate_metrics(final, gold)

            log_lines.extend([
                f"  [GOLD COUNT]:   {len(gold)} total reference citations",
                f"  [CANDIDATES]:   {hits_in_candidates} / {len(gold)} successfully mined (Top 50 Candidates)",
                f"  [SOURCE ORIGIN OF DISCOVERED GOLD]:",
                f"      -> Via Regex (Query text):   {from_regex}",
                f"      -> Via BM25 (Lexical paths): {from_bm25}",
                f"      -> Via Dense (Vector paths): {from_dense}",
                f"  [FINAL SELECTION]: {hits_in_final} / {len(gold)} retained after Reranking Filter",
                f"  [METRICS]:      Precision: {prec:.4f} | Recall: {rec:.4f} | F1-Score: {f1:.4f}"
            ])
        else:
            log_lines.append("  [METRICS]:      No gold references available (Inference Mode).")
            
        log_lines.append("="*50)
        tqdm.write("\n".join(log_lines))

    pd.DataFrame(submission_data).to_csv("submission.csv", index=False)
    print("\nPredictions saved to submission.csv")

    if has_gold:
        precisions, recalls, f1s = [], [], []
        for p, g in zip(all_preds, all_golds):
            prec, rec, f1 = calculate_metrics(p, g)
            precisions.append(prec)
            recalls.append(rec)
            f1s.append(f1)
        
        print("\n==========================================")
        print(f"Total Queries:     {len(f1s)}")
        print(f"Mean Precision:    {np.mean(precisions):.4f}")
        print(f"Mean Recall:       {np.mean(recalls):.4f}")
        print(f"Mean F1-Score:     {np.mean(f1s):.4f}")
        print("==========================================")

if __name__ == "__main__":
    main()