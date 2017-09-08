import nltk


input_file = open('./Project1/SentimentDataset/Dev/neg.txt').read() 
#print File
tokens = nltk.word_tokenize(input_file)
print tokens
