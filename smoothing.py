import preprocessing as ngrams
import math
from decimal import *
import csv

import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews

from collections import defaultdict
import preprocessing as ngrams

train_tokens = ngrams.createTokens('./Project1/SentimentDataset/Train/pos.txt');
train_tokens_neg = ngrams.createTokens('./Project1/SentimentDataset/Train/neg.txt');
dev_tokens = ngrams.createTokens('./Project1/SentimentDataset/Dev/neg.txt');

def unigramSmoothing(tokens,k=1):
        getcontext().prec = 7
        length = len(tokens)
        unidict=ngrams.build_unidict(tokens)
        uni_prob_test ={}
        for t in dev_tokens:
            if t not in uni_prob_test:
                uni_prob_test[t] = (Decimal(1.0*k) + Decimal(unidict.get(t, 0)))/ Decimal(length+(len(unidict)*k));

        return uni_prob_test

def bigramLaplaceSmoothing(train_tokens):
        getcontext().prec = 7
        length = len(train_tokens)
        unidict= ngrams.build_unidict(train_tokens)
        bidict=defaultdict(dict)
        bidict=ngrams.build_bidict(train_tokens)
        bi_prob_test = defaultdict(dict)
        x= Decimal(1.0)
        bi_len=len(bidict)
        for i in range(0,len(dev_tokens)-1):
            if dev_tokens[i] in bidict.keys():
	        if dev_tokens[i+1] in bidict[dev_tokens[i]]:
                    bi_prob_test[dev_tokens[i]][dev_tokens[i+1]] = (x + Decimal(bidict[dev_tokens[i]][dev_tokens[i+1]]))/\
									Decimal(bi_len + unidict[dev_tokens[i]])
                else:
		    z=unidict[dev_tokens[i]]
        	    y= x/Decimal(bi_len+z)
		    bi_prob_test[dev_tokens[i]][dev_tokens[i+1]] = y
	    else:
                bi_prob_test[dev_tokens[i]][dev_tokens[i+1]] = Decimal(1.0) / Decimal(len(unidict))

#        for key,value in bi_prob_test.items():
#   	    print (key, value)
        return bi_prob_test

def uni_perplexity():
    length_dev =  len(dev_tokens)
    uni_prob_dev = unigramSmoothing(train_tokens,1.2)
    logSum =0;
    for t in dev_tokens:
        logSum = Decimal(logSum + Decimal(math.log(uni_prob_dev[t])));
    print math.exp(Decimal( -logSum/length_dev))



def bi_perplexity():
    length_dev = len(dev_tokens)
    bi_prob_dev = bigramLaplaceSmoothing(train_tokens)
    logSum =0;
    for i in range(1,length_dev):
        first_word = dev_tokens[i-1];
        second_word = dev_tokens[i];
        logSum = Decimal(logSum + Decimal(math.log(bi_prob_dev[first_word][second_word])));
    print math.exp(Decimal( -logSum/length_dev))
    
def uni_classify():
    val =0
    print("abc" ,val )
    uni_prob_pos = unigramSmoothing(train_tokens,0.99)
    uni_prob_neg = unigramSmoothing(train_tokens_neg,0.99)
    i=1;
    file = open('test.csv','wt')
    writer = csv.writer(file)
    with open('./Project1/SentimentDataset/Test/test.txt') as f:
        for line in f:
            tokens = nltk.word_tokenize(line)
            stopwords = ['...', '/', '\\' , '--']
            new_words = [word for word in tokens if word not in stopwords]
            words = [w.lower() for w in new_words]
            pos_prob=0
            neg_prob=0
            for t in words:
                pos_prob += math.log(uni_prob_pos[t])
                neg_prob += math.log(uni_prob_neg[t])
            #print("positive" , -pos_prob)
            #print("negative" , -neg_prob)
            if pos_prob<neg_prob:
				writer.writerow( (i,0) )
                #file.write(i ,',',0);
            else :
                writer.writerow( (i,1) )
            i=i+1

	file.close()

def bi_classify():
    val =0
    print("abc" ,val )
    bi_prob_pos = bigramLaplaceSmoothing(train_tokens)
    bi_prob_neg = bigramLaplaceSmoothing(train_tokens_neg)
    j=1;
    file = open('test.csv','wt')
    writer = csv.writer(file)
    with open('./Project1/SentimentDataset/Test/test.txt') as f:
        for line in f:
            tokens = nltk.word_tokenize(line)
            stopwords = ['...', '/', '\\' , '--']
            new_words = [word for word in tokens if word not in stopwords]
            words = [w.lower() for w in new_words]
            pos_prob=0
            neg_prob=0
            for i in (0,len(words)-2):
                pos_prob += math.log(bi_prob_pos[words[i]][words[i+1]])
                neg_prob += math.log(bi_prob_neg[words[i]][words[i+1]])
            if pos_prob>neg_prob:
		writer.writerow( (j,0) )
            else :
                writer.writerow( (j,1) )
            j=j+1

	file.close()


#bigramLaplaceSmoothing()
#uni_perplexity()
#bi_perplexiy()
#uni_perplexity()
#unigramSmoothing()


