import preprocessing as ngrams
import random




unigram_map = ngrams.unigram_prep('./Project1/SentimentDataset/Dev/neg.txt')

print (len(unigram_map))

for x in range(10):
  ran= random.randint(0,len(unigram_map))
  print unigram_map[ran]
print



