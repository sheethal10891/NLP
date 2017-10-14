import csv
import nltk
import sklearn
import sklearn_crfsuite

inputFILE = open('./train.txt').readlines();
n=len(inputFILE)
print n

test_file= open('./test.txt').readlines()
def format_data(inputFILE): 
   list2=[]
   list1=[]
   for i in range(2,n,3):
   	tokens= inputFILE[i-2].split()
   	posTags= inputFILE[i-1].split()
   	bioTags=inputFILE[i].split()
        list1=zip(tokens,posTags,bioTags)   
        list2.append(list1)
   return list2


def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]
    
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],        
    }
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True
        
    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True
                
    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, postag, label in sent]

def sent2tokens(sent):
    return [token for token, postag, label in sent]


def train(file_name):
    train_sents=format_data(file_name)
    X_train = [sent2features(s) for s in train_sents]
    Y_train = [sent2labels(s) for s in train_sents]

    crf = sklearn_crfsuite.CRF(
       algorithm='lbfgs', 
       c1=0.1, 
       c2=0.1, 
       max_iterations=5, 
       all_possible_transitions=True)
    crf.fit(X_train, Y_train)
    labels = list(crf.classes_)
    labels.remove('O')
#    print labels

def predict(test_file):
    train(inputFILE)
    test_sents=format_data(test_file)
    X_test = [sent2features(s) for s in test_sents]
    pred=crf.predict(X_test)
    print pred

predict(test_file)
#print len(format_data(inputFILE))
