# Project 1 - part 1
Running the code
python create-sentence.py

For variations change on line 52
bigram_sentence(20,True, "The movie");

change line 8 in create-sentence.py to pick a new file for probability table for that file

uncomment line 55 and 58 to get the probability dict for unigram and bigram

#Project 1 - part 2

# Smoothing and perplexity

python smoothing.py
In smoothing.py uncomment few lines in the end to get the smoothing and perplexity for the language model you want
Change the "dev_tokens" in smoothing.py to get the values for the data you want to check perplexity against.

# Sentiment Classification
(We have not included all the models that we used for sentimental analysis.) 
Only including the unigram with k-smoothing 
uncomment uni_classify()

# Evaluating word embeddings
Not included any code here since we used external code for initial analysis

#Sentiment classification with word embeddings
(Also including only one of the classifier(simple SVN) that we teted)

python wordEmbeddings.py

We used both training and dev set together to increse the size of the training set for better results.