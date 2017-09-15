import preprocessing as ngrams
import random
import nltk

global tokens

# the file to create probability and sentence generation from
tokens = ngrams.createTokens('./Project1/SentimentDataset/Train/pos.txt');


def unigram_sentence(length):
    unigram_map = ngrams.unigram_prep(tokens)
    sentence ='';
    for x in range(length):
        ran= random.randint(0,len(unigram_map))
        sentence = sentence + " " +unigram_map[ran]
    return sentence

# length - min length of the sentence to be produced , 
# complete_sentence - Should the sentence generation continue until a period is reached
# sentence - the seed
def bigram_sentence(length,complete_sentence=False,sentence=''):
    bigram_dict= ngrams.get_bigrams(tokens)
    #bigram_prob_dict = get_biprob(bigram_dict,num_bigrams)
    start_len=len(sentence)        
    if start_len == 0:
       result= bigram_dict['.']
       start_token = result[len(result)-1]
       sentence = start_token;
    else:
       sentence_tokens = sentence.split()
       start_token = sentence_tokens[len(sentence_tokens)-1]

    while (length):
       next=ngrams.predict_next(bigram_dict,start_token) 
       sentence = sentence + ' '+ next
       length=length-1;
       start_token = next;
       
    if (complete_sentence) :
        while start_token != '.' :
            next=ngrams.predict_next(bigram_dict,start_token) 
            sentence = sentence + ' '+ next
            length=length-1;
            start_token = next; 
    return sentence    



print unigram_sentence(20)

print bigram_sentence(20,True, "The movie")

# get unigram probability table
#print ngrams.unigram_prob(ngrams.unigram_prep(tokens))

# get bigram probability table
#print ngrams.bigram_prob(ngrams.get_bigrams(tokens))

