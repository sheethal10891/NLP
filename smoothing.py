from __future__ import print_function
from decimal import *
from collections import defaultdict
import preprocessing as ngrams

train_tokens = ngrams.createTokens('./Project1/SentimentDataset/Train/pos.txt');
dev_tokens = ngrams.createTokens('./Project1/SentimentDataset/Dev/pos.txt');

def unigramLaplaceSmoothing():
        getcontext().prec = 7
        length = len(train_tokens)
        unidict=ngrams.build_unidict(train_tokens)
        uni_prob_test ={}
        for t in dev_tokens:
            uni_prob_test[t] = (Decimal(1.0) + Decimal(unidict.get(t, 0)))/ Decimal(length+len(unidict));
    
unigramLaplaceSmoothing()

def bigramLaplaceSmoothing():
        getcontext().prec = 7
        length = len(train_tokens)
        unidict= ngrams.build_unidict(train_tokens)
#        print((unidict))
        bidict=defaultdict(dict)
#        print(bidict)
        bidict=ngrams.build_bidict(train_tokens)
        bi_prob_test = defaultdict(dict)
#        print(bi_prob_test)
        x= Decimal(1.0)
        bi_len=len(bidict)
        for i in range(0,len(dev_tokens)-1):
            if dev_tokens[i] in bidict.keys():
	        if dev_tokens[i+1] in bidict[dev_tokens[i]]:
                    bi_prob_test[dev_tokens[i]][dev_tokens[i+1]] = (x + Decimal(bidict[dev_tokens[i]][dev_tokens[i+1]]))/\
									(bi_len + unidict[dev_tokens[i]])
                else:
		    z=unidict[dev_tokens[i]]
        	    y= x/Decimal(bi_len+z)
		    bi_prob_test[dev_tokens[i]][dev_tokens[i+1]] = y
	    else:
                bi_prob_test[dev_tokens[i]][dev_tokens[i+1]] = Decimal(1.0) / Decimal(len(unidict))
      
        for key,value in bi_prob_test.items():
   	    print (key, value)    
        return bi_prob_test

bigramLaplaceSmoothing()
