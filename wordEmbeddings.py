import multiprocessing
import nltk
from gensim.models.word2vec import LineSentence
from gensim.models.word2vec import Word2Vec
import numpy as np
from sklearn import svm
import pandas as pd

files_list = open('./Project1/SentimentDataset/Train/pos.txt', 'r').readlines()

files_list1 = open('./Project1/SentimentDataset/Train/pos.txt', 'r').readlines()
files_list4 = open('./Project1/SentimentDataset/Dev/pos.txt', 'r').readlines()
files_list1.extend(files_list4)


files_list2 = open('./Project1/SentimentDataset/Train/neg.txt', 'r').readlines()
files_list5 = open('./Project1/SentimentDataset/Dev/neg.txt', 'r').readlines()

files_list2.extend(files_list5)
files_list3 = open('./Project1/SentimentDataset/Test/test.txt', 'r').readlines()

files_list.extend(files_list4)
files_list.extend(files_list2)
files_list.extend(files_list3)

print len(files_list)

# todo remove unnecesary words here!
lines = [nltk.word_tokenize(sen) for sen in files_list]
#sentences = [['first', 'sentence'], ['second', 'sentence']]
model = Word2Vec(lines, min_count=1,size=50)


lines1=[nltk.word_tokenize(sen) for sen in files_list1] 

i=0;
pos_feature_array= np.zeros((len(files_list1),50))
for line in lines1 :
    data_val=np.zeros(50)
    for word in line:
        data_val = data_val+model[word]
    pos_feature_array[i]=data_val/len(line)
    i=i+1;



lines2=[nltk.word_tokenize(sen) for sen in files_list2] 
i=0;
neg_feature_array= np.zeros((len(files_list2),50))
print(len(lines2))
for line in lines2 :
    data_val=np.zeros(50)
    for word in line:
        data_val = data_val+model[word]
    neg_feature_array[i]=data_val/len(line)
    i=i+1;
    
X=np.append(pos_feature_array,neg_feature_array, axis=0);
print("X size");
print X.shape
Y=np.append(np.ones((len(pos_feature_array),1), dtype=np.int),np.zeros((len(neg_feature_array),1), dtype=np.int))
print "Y Shape"
print Y.shape

lines_test=[nltk.word_tokenize(sen) for sen in files_list3] 

i=0;
test_feature_array= np.zeros((len(files_list3),50))
for line in lines_test :
    data_val=np.zeros(50)
    for word in line:
        data_val = data_val+model[word]
    test_feature_array[i]=data_val/len(line)
    i=i+1;
    
print neg_feature_array[0]

X_test = test_feature_array


clf = svm.SVC()
clf.fit(X, Y)

yTe=clf.predict(X_test)
print yTe[1]





n = yTe.shape[0]
indices = np.array(list(range(n)))
df = pd.DataFrame(data={'Id':indices, 'Prediction':yTe})
df = df[['Id', 'Prediction']]
df.to_csv('./WordEmbeddingTest.csv', header=True, index=False)
#[model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1) for word in lines]

#occasional occasionally typical

#print model.most_similar([ 'Garland','Texas', 'Modesto'], topn=10)
#print model['raw']

'''
params = {'size': 200, 'window': 10, 'min_count': 10, 
          'workers': max(1, multiprocessing.cpu_count() - 1), 'sample': 1E-3,}
word2vec = Word2Vec(files_list, **params)

word2vec.most_similar('occasional', topn=5)
'''