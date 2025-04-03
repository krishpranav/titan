'''
@filename: tokenizer_train.py
@date: 03-04-2025
@author: Krisna Pranav
'''

from tokenizers import Tokenizer, models, trainers, pre_tokenizers

tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()

trainer = trainers.BpeTrainer(
    vocab_size=50257,
    min_frequency=2,
    special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
)

tokenizer.train(["data/raw/text_corpus.txt"], trainer)
tokenizer.save("tokenizer/moondra_tokenizer.json")
