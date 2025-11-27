import spacy
import pandas as pd
import spacy

with open("./data/slk_newscrawl_2016_1M/slk_newscrawl_2016_1M-sentences.txt", "r", encoding="utf-8") as f:
    # read lines and split at tab, keep only second column
    slk_df = pd.DataFrame(
        [line.strip().split("\t")[1] for line in f.readlines()], columns=["sentence"]
    )

# get example sized corpus
slk_df = slk_df.sample(n=100, random_state=42).reset_index(drop=True)

# Load Slovak model
nlp = spacy.load("sk_dep_web_md")

# Enable GPU processing
spacy.prefer_gpu()

# Use nlp.pipe for fast batch processing
texts = slk_df["sentence"].astype(str).tolist()

tokens = []
for doc in nlp.pipe(texts, batch_size=50):
    tokens.append([token.text.lower() for token in doc])

slk_df["tokens"] = tokens
slk_df.to_csv("./data/slk_preprocessed_example.csv", index=False)