import pandas as pd
from collections import defaultdict
import numpy as np
from sklearn.model_selection import train_test_split
import ast
import nltk
from nltk.util import ngrams, everygrams, pad_sequence # Added everygrams
from nltk.lm import MLE, Lidstone, Vocabulary, KneserNeyInterpolated 
from itertools import chain 

N = 3 # N-gram size
TEST_SIZE = 0.10 
UNK_TOKEN = '<UNK>' # OOV word token
CSV_FILE = "../data/preprocessed/slk_tokenized.csv"

with open(CSV_FILE, 'r', encoding='utf-8') as f:
    df = pd.read_csv(f)

train_sentences, test_sentences = train_test_split(
    df['tokens'].tolist(),
    test_size=TEST_SIZE,
    random_state=42
)

train_corpus_flat = [token for sentence in train_sentences for token in sentence]
train_vocab = sorted(list(set(train_corpus_flat)))

# Create the NLTK Vocabulary object
full_vocab_for_model = Vocabulary(train_vocab, unk_label=UNK_TOKEN)

new_test_sentences = []
for sentence in test_sentences:
    unk_sentence = [full_vocab_for_model.lookup(token) for token in sentence]
    new_test_sentences.append(unk_sentence)

test_sentences = new_test_sentences

train_data = []
for tokens in train_sentences:
    padded = list(pad_sequence(tokens, n=N, 
                               pad_left=True, left_pad_symbol="<s>", 
                               pad_right=True, right_pad_symbol="</s>"))
    
    sent_ngrams = list(everygrams(padded, max_len=N))
    train_data.append(sent_ngrams)

lm = Lidstone(gamma=1, order=N)
lm.fit(train_data, full_vocab_for_model)

# Prepare the test data by padding
padded_test = [
    list(pad_sequence(tokens, n=N, 
                      pad_left=True, left_pad_symbol="<s>",
                      pad_right=True, right_pad_symbol="</s>"))
    for tokens in test_sentences
]

test_tokens_flat = [token for sent in padded_test for token in sent]

# Calculate perplexity on the test corpus
corpus_pp = lm.perplexity(ngrams(test_tokens_flat, N))

print(corpus_pp)