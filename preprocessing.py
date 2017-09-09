import nltk


def createTokens(path):
    print("abc")
    lines = open(path).read()
    tokens = nltk.word_tokenize(lines)
    words = [w.lower() for w in tokens]
    return words

def unigram_prep(path):
    words = createTokens(path);
    #print(words)
    return sorted(words)
    
#print(unigram_prep('./Project1/SentimentDataset/Dev/neg.txt'))