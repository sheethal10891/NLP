import nltk
import random

def createTokens(path):
    lines = open(path).read()
    tokens = nltk.word_tokenize(lines)
    words = [w.lower() for w in tokens]
    return words

def unigram_prep(words):
    
    #print(words)
    return sorted(words)

def probability_calc(tokens,dictionary,total_tokens):
    prob_dict={}
    for token in tokens:
        prob_dict[token]=dictionary[token]/total_tokens
    return prob_dict

#Get  the next probable word for a token
def predict_next(bigram_dict,start_word):
    ##TODO: write logic to get the word with max probability
    word_map = bigram_dict[start_word]
    #print(len(word_map))
    rn=random.randint(0,len(word_map)-1)
    #print(" Random")
    #print(rn)
    word=word_map[rn]
    return word

def get_biprob(bigram_dict,num_bigrams):
     prob_dict={}
     for token in bigram_dict.keys():
        for  next in bigram_dict[token].keys():
	    prob_dict[token][next] = bigram[token][next] /len(bigram[token].keys())    
     return prob_dict

## Build dictionary of bigrams  
def get_bigrams(tokens):
    bigram_dict={}
    for i in range(0,len(tokens)-1):
            bigram_dict.setdefault(tokens[i],[]).append(tokens[i+1])
    return bigram_dict

## Random sentence generator bigrams 
## Input: max len of sentence,Initial unigram tokens,optional sentence start tokens

