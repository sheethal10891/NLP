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
dev_tokens = ngrams.createTokens('./Project1/SentimentDataset/Test/test.txt');
#dev_tokens = ngrams.createTokens('./Project1/SentimentDataset/Dev/pos.txt');
#dev_tokens = ngrams.createTokens('./Project1/SentimentDataset/Dev/neg.txt');


## LAPLACE && K_SMOOTHING
def unigramLaplaceSmoothing(tokens,k=1):
        getcontext().prec = 7
        length = len(tokens)
        unidict=ngrams.build_unidict(tokens)
        uni_prob_test ={}
        for t in dev_tokens:
            if t not in uni_prob_test:
                uni_prob_test[t] = (Decimal(1.0*k) + Decimal(unidict.get(t, 0)))/ Decimal(length+(len(unidict)*k));

        return uni_prob_test

def bigramLaplaceSmoothing(train_tokens,k=1):
        getcontext().prec = 7
        length = len(train_tokens)
        unidict= ngrams.build_unidict(train_tokens)
        bidict=defaultdict(dict)
        bidict=ngrams.build_bidict(train_tokens)
        bi_prob_test = defaultdict(dict)
        x= Decimal(1.0*k)
        bi_len=len(bidict)
        for i in range(0,len(dev_tokens)-1):
            if dev_tokens[i] in bidict.keys():
	        if dev_tokens[i+1] in bidict[dev_tokens[i]]:
                    bi_prob_test[dev_tokens[i]][dev_tokens[i+1]] = (x + Decimal(bidict[dev_tokens[i]][dev_tokens[i+1]]))/\
									Decimal(bi_len*k + unidict[dev_tokens[i]])
                else:
		    z=unidict[dev_tokens[i]]
        	    y= x/Decimal((bi_len*k+z))
		    bi_prob_test[dev_tokens[i]][dev_tokens[i+1]] = y
	    else:
                bi_prob_test[dev_tokens[i]][dev_tokens[i+1]] = Decimal(1.0) / Decimal(len(unidict)*k)

#        for key,value in bi_prob_test.items():
#   	    print (key, value)
        return bi_prob_test

##PERPLEXITY
def uni_perplexity():
    length_dev =  len(dev_tokens)
    uni_prob_dev = unigramLaplaceSmoothing(train_tokens,0.6)
    logSum =0;
    for t in dev_tokens:
        logSum = Decimal(logSum + Decimal(math.log(uni_prob_dev[t])));
    print math.exp(Decimal( -logSum/length_dev))

def bi_perplexity():
    length_dev = len(dev_tokens)
    bi_prob_dev = bigramLaplaceSmoothing(train_tokens,0.01)
    logSum =0;
    for i in range(1,length_dev):
        first_word = dev_tokens[i-1];
        second_word = dev_tokens[i];
        logSum = Decimal(logSum + Decimal(math.log(bi_prob_dev[first_word][second_word])));
    print math.exp(Decimal( -logSum/length_dev))
    
##KNESER-NEY

def KNSmoothing(train_tokens,delta=1):
        getcontext().prec = 7
        length = len(train_tokens)
        unidict= ngrams.build_unidict(train_tokens)
        bidict=defaultdict(dict)
        bidict=ngrams.build_bidict(train_tokens)
        x= Decimal(1.0*delta)
        bi_len=len(bidict)
        bi_prob_test=defaultdict(dict)
        
        for i in range(0,len(dev_tokens)-1):
            t1= dev_tokens[i]
            t2=dev_token[i+1]
            if t1 in bidict.keys() and t2 in bidict[t1]:
                if bidict[t1][t2] > 2:
                    numerator = bidict[t1][t2] - 0.75
                elif bidict[t1][t2]==1:
                    numerator = bidict[t1][t2] -0.5

                if t1 in unidict.keys():
                    denominator = unidict[t1]
                else:
                    denominator = unidict['UNK']
            else:
                numerator = 1
                denominator = len(unidict)
            bi_prob_test[t1][t2]=numerator/denominator

#        for key,value in bi_prob_test.items():
#   	    print (key, value)
        return bi_prob_test

def perplexityUsingPositiveBigrams(string1):
    #string1 = '<start> ' + string1 + ' <end>'
    string1 = string1.translate(str.maketrans('','',string.punctuation))
    
    string1 = '<start> ' + string1
    strList = string1.split()
    n = len(strList)-1
    
    perplexityValue = 0
    
    #unseen = false
    for i in range(1, len(strList)):
        #print(strList[i-1],':', strList[i])
        k1 = strList[i-1]
        k2 = strList[i]
        numerator = 1
        denominator = 1
        if k1 in positive_bigramCounts and k2 in positive_bigramCounts[k1]:
            if positive_bigramCounts[k1][k2] > 2:
                numerator = positive_bigramCounts[k1][k2] - 0.75
            elif positive_bigramCounts[k1][k2] == 1:
                numerator = positive_bigramCounts[k1][k2] - 0.5
            
            if k1 in positive_unigramCounts:
                denominator = positive_unigramCounts[k1]
            else:
                denominator = positive_unigramCounts['unk']
            perplexityValue += math.log(numerator/denominator)
        else:
            perplexityValue += math.log(positive_ksmoothing(k1, k2))



def uni_classify():
    val =0
    print("abc" ,val )
    uni_prob_pos = unigramLaplaceSmoothing(train_tokens,0.9)
    uni_prob_neg = unigramLaplaceSmoothing(train_tokens_neg,0.9)
    i=1;
    file = open('unitest.csv','wt')
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
#            print("positive" , -pos_prob)
#            print("negative" , -neg_prob)
            if pos_prob>neg_prob:
		writer.writerow( (i,0) )
            else :
                writer.writerow( (i,1) )
            i=i+1

	file.close()

def bi_classify():
    val =0
    print("abc" ,val )
    bi_prob_pos = bigramLaplaceSmoothing(train_tokens,0.9)
    bi_prob_neg = bigramLaplaceSmoothing(train_tokens_neg,0.9)
    j=1;
    file = open('bitest.csv','wt')
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


bi_classify()
#uni_classify()
#uni_perplexity()
#bi_perplexity()


