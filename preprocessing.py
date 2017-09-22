import nltk
import random
from decimal import *

from collections import Counter
from collections import defaultdict

def createTokens(path):
    lines = open(path).read()
    tokens = nltk.word_tokenize(lines)
    stopwords = ['...', '/', '\\' , '--']
    # remove unwanted tokens and add start and end sentence tokens
    new_words = [word for word in tokens if word not in stopwords]
    words = [w.lower() for w in new_words]
    return words

def unigram_prep(words):
    
    #print(words)
    return sorted(words)

def build_unidict(tokens):
    uni_dict={"UNK":0}
    length = len(tokens)
    for token in tokens:
        if token not in uni_dict:
            uni_dict[token] = 1
        else:
            uni_dict[token] += 1
    
    return uni_dict

def add_unknowns(idict):
    count =0  
    for token in idict:
        if (idict[token] == 1):
         count+=1 
    idict['UNK']=count

def unigram_prob(tokens):
    length = len(tokens)
    getcontext().prec = 7
    uni_dict=unigram_dict(tokens)
    uni_prob ={}
    for t in uni_dict:
        uni_prob[t] = Decimal(uni_dict[t])/Decimal(length)
    return uni_prob
    
    
def bigram_prob(bi_dict):
    getcontext().prec = 7
    bi_prob = {}
    for t in bi_dict :
        bi_prob[t] ={}
        for word in bi_dict[t]:
            if word not in bi_prob[t]:
                bi_prob[t][word] = 1
            else:
                bi_prob[t][word] += 1
                  
    for token in bi_prob:
        for w in bi_prob[token]:
            bi_prob[token][w] = Decimal(bi_prob[token][w])/Decimal(len(bi_prob[token]))
            
    return bi_prob
        

#Get  the next probable word for a token
def predict_next(bigram_dict,start_word):
    word_map = bigram_dict[start_word]
    rn=random.randint(0,len(word_map)-1)
    word=word_map[rn]
    return word

## Build dictionary of bigrams  
def build_bidict(tokens):
    bigram_dict= defaultdict(dict)
    for i in range(0,len(tokens)-1):
#	if tokens[i] in bigram_dict.keys():
           if tokens[i+1] in bigram_dict[tokens[i]]:
	       bigram_dict[tokens[i]][tokens[i+1]]+=1;
	   else:
	       bigram_dict[tokens[i]][tokens[i+1]]=1;
         		
#    print bigram_dict
    return bigram_dict

def get_bigrams(tokens):         
      bigram_dict={}           
      for i in range(0,len(tokens)-1):           
             bigram_dict.setdefault(tokens[i],[]).append(tokens[i+1])       
      return bigram_dict
## Random sentence generator bigrams 
## Input: max len of sentence,Initial unigram tokens,optional sentence start tokens

