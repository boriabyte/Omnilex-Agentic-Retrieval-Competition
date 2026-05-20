import os
import re
from collections import Counter

import pandas as pd
from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
LOG_FILE = os.path.join(LOG_DIR, "data_exploration.txt")

FILENAMES = {
    "court_considerations": "court_considerations.csv",
    "laws_de": "laws_de.csv",
}


def load_csv(name, out):
    path = os.path.join(DATA_DIR, FILENAMES[name])

    out.write(f"\nloading {name}: {path}\n")

    try:
        df = pd.read_csv(path)
        out.write(f"rows: {len(df)}\n")
        return df

    except Exception as e:
        out.write(f"failed: {e}\n")
        return pd.DataFrame()


def detect_languages(df, sample_size=500):
    texts = df["text"].dropna().astype(str).head(sample_size)

    langs = []

    for t in texts:
        try:
            langs.append(detect(t))
        except:
            langs.append("err")

    return Counter(langs)


def has_article_structure(x):
    x = str(x).strip()

    patterns = [
        r"^Art\.\s*\d+",
        r"^§\s*\d+",
        r"^Artikel\s+\d+",
    ]

    return any(re.search(p, x) for p in patterns)


def analyze_duplicates(df, out):
    out.write("\n--- duplicate analysis (text vs text+title) ---\n")

    n = len(df)
    text = df["text"].fillna("").astype(str).str.strip()

    has_title = "title" in df.columns
    if has_title:
        title = df["title"].fillna("").astype(str).str.strip()
    else:
        title = pd.Series([""] * n, index=df.index)
        out.write("(no title column -- text+title == text for this dataset)\n")

    n_unique_text = text.nunique()
    n_dupe_text_rows = n - n_unique_text          
    out.write(f"rows total:                  {n}\n")
    out.write(f"unique by text:              {n_unique_text}\n")
    out.write(f"duplicate rows by text:      {n_dupe_text_rows}\n")

    pair = pd.DataFrame({"text": text, "title": title})
    n_unique_pair = len(pair.drop_duplicates())
    n_dupe_pair_rows = n - n_unique_pair
    out.write(f"unique by text+title:        {n_unique_pair}\n")
    out.write(f"duplicate rows by text+title:{n_dupe_pair_rows}\n")

    genuine_dupe_rows = n_dupe_pair_rows
    title_distinguished_rows = n_dupe_text_rows - n_dupe_pair_rows

    out.write("\nof the duplicate-by-text rows:\n")
    out.write(f"  genuine (same text & same title):       "
              f"{genuine_dupe_rows}\n")
    out.write(f"  title-distinguished (same text, diff title): "
              f"{title_distinguished_rows}\n")

    if n_dupe_text_rows > 0:
        pct = 100.0 * title_distinguished_rows / n_dupe_text_rows
        out.write(f"  -> {pct:.1f}% of text-duplicates are only "
                  f"separated by their title\n")

    if has_title:
        titles_per_text = pair.groupby("text")["title"].nunique()
        shared = titles_per_text[titles_per_text > 1]
        out.write(f"\ndistinct texts appearing under >1 title: "
                  f"{len(shared)}\n")
        if len(shared):
            out.write(f"  max titles for one text: {shared.max()}\n")
            out.write("  top shared texts (title count -- text):\n")
            for t in shared.sort_values(ascending=False).head(10).index:
                cnt = titles_per_text[t]
                out.write(f"    [{cnt:>4} titles] {t[:90]}\n")


def inspect_dataset(name, df, out):
    out.write(f"\n==================== {name} ====================\n")

    if df.empty:
        out.write("empty dataframe\n")
        return

    out.write(f"rows: {len(df)}\n")
    out.write(f"columns: {list(df.columns)}\n")

    if "text" in df.columns:
        text = df["text"].fillna("").astype(str)

        lengths = text.str.len()

        too_short = lengths < 20

        single_word = text.str.split().str.len() == 1

        ellipsis = text.str.strip().isin(["...", "(...)", "…"])

        truncated = (
            (lengths > 40)
            & (~text.str.endswith((".", "!", "?", ";", ":")))
        )

        out.write(f"too short: {too_short.sum()}\n")
        out.write(f"single word: {single_word.sum()}\n")
        out.write(f"ellipsis only: {ellipsis.sum()}\n")
        out.write(f"possibly truncated: {truncated.sum()}\n")

        out.write("\n--- text length ---\n")
        out.write(f"mean: {lengths.mean():.2f}\n")
        out.write(f"median: {lengths.median():.2f}\n")
        out.write(f"max: {lengths.max()}\n")
        out.write(f"min: {lengths.min()}\n")

        for threshold in [500, 1000, 2000, 4000, 8000]:
            cnt = (lengths > threshold).sum()

            out.write(
                f">{threshold}: {cnt} ({cnt/len(df):.2%})\n"
            )

        out.write("\n--- duplicates ---\n")

        dupes = text.duplicated().sum()

        out.write(f"duplicate texts: {dupes}\n")
        out.write(f"unique texts: {text.nunique()}\n")

        analyze_duplicates(df, out)

        out.write("\n--- shitty text ---\n")

        shitty = text.apply(is_shitty_text)

        out.write(
            f"shitty rows: "
            f"{shitty.sum()} ({shitty.mean():.2%})\n"
        )

        out.write("\nshitty text samples:\n")

        samples = text[shitty].head(30)

        for s in samples:
            out.write("\n-----------------\n")
            out.write(f"{s[:1000]}\n")

        if "citation" in df.columns:
            citation_stats = (
                df.groupby("text")["citation"]
                .nunique()
            )

            out.write(
                f"mean citations/text: "
                f"{citation_stats.mean():.2f}\n"
            )

            out.write(
                f"median citations/text: "
                f"{citation_stats.median():.2f}\n"
            )

            out.write(
                f"max citations/text: "
                f"{citation_stats.max()}\n"
            )

        out.write("\n--- languages ---\n")

        langs = detect_languages(df)

        total = sum(langs.values())

        for lang, cnt in langs.most_common():
            out.write(
                f"{lang}: {cnt} ({cnt/total:.2%})\n"
            )

    if name == "laws_de" and "citation" in df.columns:
        citations = df["citation"].fillna("").astype(str)

        structured = citations.apply(has_article_structure)

        valid_cnt = structured.sum()
        invalid_cnt = (~structured).sum()

        out.write("\n--- citation structure (laws_de only) ---\n")

        out.write(
            f"valid article citations: "
            f"{valid_cnt} ({valid_cnt/len(df):.2%})\n"
        )

        out.write(
            f"invalid article citations: "
            f"{invalid_cnt} ({invalid_cnt/len(df):.2%})\n"
        )

        out.write("\ninvalid citation samples:\n")

        invalid_samples = (
            citations[~structured]
            .drop_duplicates()
            .head(50)
        )

        for x in invalid_samples:
            out.write(f"- {x}\n")


def is_shitty_text(x):
    x = str(x).strip()

    if not x:
        return True

    if len(x) < 5:
        return True

    if x == "...":
        return True

    if x == "(...)":
        return True

    if x == "…":
        return True

    if re.fullmatch(r"[\W\d_]+", x):
        return True

    words = x.split()

    if len(words) == 1 and len(words[0]) < 4:
        return True

    if "�" in x:
        return True

    if len(x) > 40 and not re.search(r"[.!?;:]$", x):
        last = words[-1].lower()

        bad_last = {
            "und",
            "oder",
            "mit",
            "kein",
            "eine",
            "der",
            "die",
            "das",
        }

        if last in bad_last:
            return True

    return False

def main():
    os.makedirs(LOG_DIR, exist_ok=True)

    with open(LOG_FILE, "w", encoding="utf-8") as out:

        for name in FILENAMES:
            df = load_csv(name, out)
            inspect_dataset(name, df, out)

    print(f"saved -> {LOG_FILE}")


if __name__ == "__main__":
    main()