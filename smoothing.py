import preprocessing as ngrams
from decimal import *


train_tokens = ngrams.createTokens('./Project1/SentimentDataset/Train/pos.txt');
dev_tokens = ngrams.createTokens('./Project1/SentimentDataset/Train/pos.txt');

def unigramLaplaceSmoothing():
        getcontext().prec = 7
        length = len(tokens)
    	unidict=ngrams.unigram_dict(train_tokens)
    	bidict=ngrams.get_bigrams(train_tokens)
        uni_prob_test ={}
        for t in dev_tokens:
            if t not in uni_prob_test:
      			uni_prob_test[t] = (1.0 + Decimal(unidict[t]))/ Decimal(2*length);
    