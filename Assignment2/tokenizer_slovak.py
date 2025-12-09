from tokenizers import Tokenizer
from tokenizers import models, trainers, normalizers, pre_tokenizers, processors, decoders
from collections import Counter
import pandas as pd
import random
import re
import torch

with open(r"data/slk_newscrawl_2016_1M/slk_newscrawl_2016_1M-sentences.txt", "r", encoding="utf-8") as f:
    slk_df = pd.DataFrame(
            [line.strip().split("\t")[1] for line in f.readlines()], columns=["sentence"]
        )

def produce_clean_sentence_list(df):
    cleaned_sentences = []
    for sentence in df["sentence"]:
        cleaned = re.sub(r'\d+', '[NUM]', sentence)
        cleaned = ''.join(char for char in cleaned if char.isalpha() or char.isspace() or char == '[' or char == ']')
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        cleaned_sentences.append(cleaned)
    return cleaned_sentences

slk_df["cleaned"] = produce_clean_sentence_list(slk_df)
sentences = slk_df["cleaned"].astype(str).tolist()

random.seed(19191)
random.shuffle(sentences)
split_index = int(0.9 * len(sentences))
train_sentences = sentences[:split_index]
test_sentences = sentences[split_index:]

print("training tokenizer...")
tokenizer = Tokenizer(models.BPE(unk_token="[UNK]", byte_fallback=True))
tokenizer.normalizer = normalizers.Sequence([
    normalizers.NFC(),  
    normalizers.Lowercase()
])
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
tokenizer.decoder = decoders.ByteLevel()
tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)

special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "[NUM]"]
trainer = trainers.BpeTrainer(
    vocab_size=50_000,
    min_frequency=2,
    special_tokens=special_tokens,
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
)

tokenizer.train_from_iterator(train_sentences, trainer=trainer)

tokenizer.enable_padding(
    pad_id=tokenizer.token_to_id("[PAD]"),
    pad_token="[PAD]"
)
tokenizer.enable_truncation(max_length=64)

tokenizer.save("slovak_tokenizer.json")
print("saved slovak_tokenizer.json")

tokenizer.enable_padding(pad_id=tokenizer.token_to_id("[PAD]"), pad_token="[PAD]")
tokenizer.no_truncation()

encoded_batch = tokenizer.encode_batch(sentences)

full_token_ids = []
for output in encoded_batch:
    full_token_ids.extend(output.ids)

print(f"saving {len(full_token_ids)} tokens to train_data.pt...")
data_tensor = torch.tensor(full_token_ids, dtype=torch.long)
torch.save(data_tensor, "train_data.pt")
print("done.")