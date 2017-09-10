import preprocessing as ngrams
import random
import nltk

global tokens
tokens = ngrams.createTokens('./Project1/SentimentDataset/Dev/neg.txt');


def unigram_sentence(length):
    
    unigram_map = ngrams.unigram_prep(tokens)
    sentence ='';
    for x in range(length):
        ran= random.randint(0,len(unigram_map))
        sentence = sentence + " " +unigram_map[ran]
    return sentence
   
def bigram_sentence(length,complete_sentence=False,sentence=''):
    bigram_dict= ngrams.get_bigrams(tokens)
    #bigram_prob_dict = get_biprob(bigram_dict,num_bigrams)
    start_len=len(sentence)        

    #if no start token given,choose 1 word from the words occuring after '.'(Possible sentence start words)
    if start_len == 0:
       result= bigram_dict['.']
       #sentence = next(iter(bigram_dict))
       start_token = result[len(result)-1]
       sentence = start_token;
    else:
       sentence_tokens = sentence.split()
       start_token = sentence_tokens[len(sentence_tokens)-1]
   #below code is fwhen we use dict from trained corpus
   # if start_token not in bigram_dict:
   #    start_token = random.choice(bigram_dict.keys()) 
        
        
   #predict next_words for max count 
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

print bigram_sentence(20, True, "I am who I am")

