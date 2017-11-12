import sys
import time
import glob
import unicodedata
from konlpy.tag import Mecab
from gensim.models import Word2Vec

WINDOW = 5
EMBEDDING_SIZE = 200
BATCH_SIZE = 10000
ITER = 10

def read_text(fin):
    corpus_li = []
    mecab =
