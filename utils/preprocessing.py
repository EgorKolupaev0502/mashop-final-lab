import re

from natasha import Segmenter, MorphVocab, NewsEmbedding, NewsMorphTagger, Doc
from nltk import SnowballStemmer


def is_stopword(word):
    if len(word) < 3:
        return True
    if not re.match(r'\w+', word):
        return True
    return False


segmenter = Segmenter()
morph_vocab = MorphVocab()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)

stemmer = SnowballStemmer("russian")


def replace_numbers(token):
    if re.match(r'[А-яЁёA-z]+', token):
        return token
    if re.search(r'[\d+.,]+', token) is not None:
        return "[digits]"
    return token


def is_number(token):
    return re.search(r'[\d+.,]+', token) is not None


def stem_token(token):
    return stemmer.stem(token)


def preprocess_sentence(sentence,
                        lemmatize=False,
                        min_word_len=0,
                        stem_words=False,
                        drop_digits=True):
    doc = Doc(sentence)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)
    for token in doc.tokens:
        token.lemmatize(morph_vocab)
    if lemmatize:
        tokens = [token.lemma for token in doc.tokens]
    else:
        tokens = [token.text for token in doc.tokens]
    tokens = [token for token in tokens if len(token) >= min_word_len]
    tokens = [token for token in tokens if not is_stopword(token)]
    if stem_words:
        tokens = [stem_token(token) for token in tokens]
    if drop_digits:
        tokens = [token for token in tokens if not is_number(token)]
    else:
        tokens = [replace_numbers(token) for token in tokens]
    return tokens
