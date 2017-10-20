import csv
import sklearn
import sklearn_crfsuite
import scipy.stats
from decimal import *
from sklearn.metrics import make_scorer
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import RandomizedSearchCV
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
#from Collections import Counter

inputFILE = open('./train.txt').readlines();
n=len(inputFILE)
print n
pTrain=inputFILE[:29400]
pVal=inputFILE[29400:] 
tags5=["B-LOC","B-PER","B-ORG","B-MISC", "I-LOC","I-PER","I-ORG","I-MISC"]

tags3=["PER","LOC","MISC","ORG"]

test_file= open('./test.txt').readlines()
def format_data(inputFILE): 
   list2=[]
   list1=[]
   for i in range(2,len(inputFILE),3):
   	tokens= inputFILE[i-2].split()
   	posTags= inputFILE[i-1].split()
   	bioTags=inputFILE[i].split()
        for i in range(len(bioTags)):
            bioTags[i]=find_tag(bioTags[i])
        list1=zip(tokens,posTags,bioTags)   
        list2.append(list1)
   return list2

m=len(test_file)

def format_datatest(inputFILE1): 
   list2=[]
   list1=[]
   for i in range(2,m,3):
   	tokens= inputFILE1[i-2].split()
   	posTags= inputFILE1[i-1].split()
        bioTags= inputFILE1[i].split()
        list1=zip(tokens,posTags)   
        list2.append(list1)
   return list2

def find_tag(tag):
   if tag in (['B-PER','I-PER']):
      return 'PER'
   elif tag in (['B-MISC','I-MISC']):
      return 'MISC'
   elif tag in (['B-ORG','I-ORG']):
      return 'ORG'
   elif tag in (['B-LOC','I-LOC']):
      return 'LOC'
   else: 
      return 'O'


def createblocks(start,end):
    #print str(start)+"-"+str(end)
    return str(start)+"-"+str(end)



def createsubmission(tagged):
    finalSub={}
    finalSub['PER']=[]
    finalSub['ORG']=[]
    finalSub['MISC']=[]
    finalSub['LOC']=[]
    finalSub['O']=[]
    count =0
    for i in range(len(tagged)):
        next=tagged[i][0]
        start=count;
        senstart=count
        for j in range(len(tagged[i])):
            if next!=tagged[i][j]:
                finalSub[next].append(createblocks(start,senstart+(j-1)))
                start=senstart+j
                next=tagged[i][j]
                #print next
                #print start
            count+=1
        finalSub[next].append(createblocks(start,count-1))
        
        #break;
        #print tagged[i]
    print count
    with open('crf.csv','wt') as file:
        w = csv.writer(file,delimiter=" ")
        w.writerow(["Type","Prediction"])
        for key, value in finalSub.items():
           if key !='O': 
            w.writerow([key]+ value)
        file.close()

'''
def createsubmission(tagged):
    finalSub={}
    finalSub['PER']=[]
    finalSub['ORG']=[]
    finalSub['MISC']=[]
    finalSub['LOC']=[]
    finalSub['O']=[]
    count =0
    for i in range(len(tagged)):
        next=find_tag(tagged[i][0])
        start=count;
        senstart=count
        for j in range(len(tagged[i])):
            tagged[i][j]=find_tag(tagged[i][j])
	    if next!=tagged[i][j]:
               	     finalSub[next].append(createblocks(start,senstart+(j-1)))
                     start=senstart+j
                     next=find_tag(tagged[i][j])
            count+=1
        finalSub[next].append(createblocks(start,count-1))
    with open('crf.csv','wt') as file:
        w = csv.writer(file,delimiter=" ")
        w.writerow(["Type","Prediction"])
        for key,value in finalSub.items():
           if key !='O': 
              w.writerow([key]+ value)
        file.close()
'''

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
    return X_train,Y_train

def predict(test_file):
    X_train,Y_train=train(inputFILE)
    crf = sklearn_crfsuite.CRF(
	       algorithm='l2sgd',
               c2=0.1, 
	       max_iterations=150, 
	       all_possible_transitions=True)
    crf.fit(X_train, Y_train)
    test_sents=format_datatest(test_file)
    X_test = [sent2features(s) for s in test_sents]
    pred=crf.predict(X_test)
    createsubmission(pred)
#    print pred

#predict(test_file)

def prf_scores(pTrain,pVal):
    X_train,Y_train = train(pTrain)
    crf = sklearn_crfsuite.CRF(
	       algorithm='pa',
#               c1=0.1, 
#	       c2=0.1, 
	       max_iterations=100, 
	       all_possible_transitions=True)
    crf.fit(X_train, Y_train)
    test_sents = format_data(pVal)
    X_test = [sent2features(s) for s in test_sents]
    pred= crf.predict(X_test)
    nVal=len(pVal)
    correct=0
    predicted=0
    totalTags=0
    k=0

    for i in range(2,nVal,3):
        words=pVal[i-2].split()
        biotags=pVal[i].split()
        for j in range(len(words)):
            if biotags[j] in tags5:
                totalTags+=1
                if pred[k][j] in tags3:
                    predicted+=1
                    tag=pred[k][j]
                    if tag==find_tag(biotags[j]):
                        correct+=1
        k+=1                
    print correct
    print predicted
    print totalTags
    precision=Decimal(correct)/Decimal(predicted)
    recall=Decimal(correct)/Decimal(totalTags)
    fMeasure=2*precision*recall/(precision+recall)
    print precision
    print recall
    print fMeasure

#prf_scores(pTrain,pVal)    
   
'''
    
    
def predict(test_file):
        X_train,Y_train=train(inputFILE)
	crf=sklearn_crfsuite.CRF(
	    algorithm='lbfgs', 
	    max_iterations=100, 
	    all_possible_transitions=True
	)
	params_space = {
	    'c1': scipy.stats.expon(scale=0.5),
	    'c2': scipy.stats.expon(scale=0.05),
	}
        labels=['B-LOC', 'B-ORG', 'B-PER', 'I-PER', 'B-MISC', 'I-ORG', 'I-LOC', 'I-MISC']
	# use the same metric for evaluation
	f1_scorer = make_scorer(metrics.flat_f1_score, 
				average='weighted', labels=labels)

	# search
	rs = RandomizedSearchCV(crf, params_space, 
				cv=3, 
				verbose=1, 
				n_jobs=-1, 
				n_iter=10, 
				scoring=f1_scorer)
	rs.fit(X_train, Y_train)
        
	print('best params:', rs.best_params_)
	print('best CV score:', rs.best_score_)
	print('model size: {:0.2f}M'.format(rs.best_estimator_.size_ / 1000000))
'''
