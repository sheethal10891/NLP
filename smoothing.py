import preprocessing as ngrams
import math
from decimal import *
import csv

import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews


train_tokens = ngrams.createTokens('./Project1/SentimentDataset/Train/pos.txt');
train_tokens_neg = ngrams.createTokens('./Project1/SentimentDataset/Train/neg.txt');
dev_tokens = ngrams.createTokens('./Project1/SentimentDataset/Test/test.txt');

def unigramLaplaceSmoothing(tokens,k=1):
        getcontext().prec = 7
        length = len(tokens)
        unidict=ngrams.unigram_dict(tokens)
        uni_prob_test ={}
        for t in dev_tokens:
            if t not in uni_prob_test:
                uni_prob_test[t] = (Decimal(1.0*k) + Decimal(unidict.get(t, 0)))/ Decimal(length+(len(unidict)*k));
                
        return uni_prob_test

def uni_perplexity():
    length_dev =  len(dev_tokens)
    uni_prob_dev = unigramLaplaceSmoothing(train_tokens,0.99)
    logSum =0;
    for t in dev_tokens:
        logSum = Decimal(logSum + Decimal(math.log(uni_prob_dev[t])));
    print math.exp(Decimal( -logSum/length_dev))
    


def bi_perplexiy():
    length_dev = len(dev_tokens)
    bi_prob_dev = unigramLaplaceSmoothing()
    logSum =0;
    for i in range(1,length_dev):
        first_word = dev_tokens[i-1];
        second_word = dev_tokens[i];
        logSum = Decimal(logSum + Decimal(math.log(bi_prob_dev[first_word][second_word])));
    print math.exp(Decimal( -logSum/length_dev))
    
def classify():
    val =0
    print("abc" ,val )
    uni_prob_pos = unigramLaplaceSmoothing(train_tokens,0.99)
    uni_prob_neg = unigramLaplaceSmoothing(train_tokens_neg,0.99)
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
            print("positive" , -pos_prob)
            print("negative" , -neg_prob)
            if pos_prob<neg_prob:
				writer.writerow( (i,0) )
                #file.write(i ,',',0);
            else :
                writer.writerow( (i,1) )
            i=i+1
			
	file.close()
            
classify()        	
    
        
