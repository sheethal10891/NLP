import preprocessing as ngrams
from decimal import *


train_tokens = ngrams.createTokens('./Project1/SentimentDataset/Train/pos.txt');
dev_tokens = ngrams.createTokens('./Project1/SentimentDataset/Dev/pos.txt');

def unigramLaplaceSmoothing():
        getcontext().prec = 7
        length = len(train_tokens)
        unidict=ngrams.unigram_dict(train_tokens)
        uni_prob_test ={}
        for t in dev_tokens:
            if t not in uni_prob_test:
                uni_prob_test[t] = (Decimal(1.0) + Decimal(unidict.get(t, 0)))/ Decimal(2*length);
                
        #print(uni_prob_test)
    
    
unigramLaplaceSmoothing()
