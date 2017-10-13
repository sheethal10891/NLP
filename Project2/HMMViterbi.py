import csv
import random
from decimal import *
from collections import defaultdict


tags=["B-LOC","B-PER","B-ORG","B-MISC", "I-LOC","I-PER","I-ORG","I-MISC","O"]
tags2=["PER","LOC","MISC","ORG","O"]

tags3=["PER","LOC","MISC","ORG"]
#tags2=["PER","LOC","MISC","ORG"]


alllines = open('./train.txt').readlines();
testSet=open('./test.txt').readlines();


def getEmissionProb(train):
    emprob={}
    emcount={}
    n=len(train)
    for i in xrange(2,n,3):
        biotags=train[i].split()
        words=train[i-2].split()
        for j in range(len(biotags)):
            if biotags[j][2:] in tags2:
            	tag=biotags[j][2:]
            else :
                tag="O"
            if words[j] not in emcount:
                emcount[words[j]]={}
                emprob[words[j]]={}
                emcount[words[j]]["count"]=0
            if tag in emcount[words[j]]:
            	emcount[words[j]][tag]+=1
                emcount[words[j]]["count"]+=1
                emprob[words[j]][tag]=Decimal(emcount[words[j]][tag])/Decimal(emcount[words[j]]["count"])
            else:
               	emcount[words[j]][tag]=1
                emcount[words[j]]["count"]+=1
                #print  Decimal(1)/Decimal(emcount[words[j]]["count"])
                emprob[words[j]][tag]=Decimal(1)/Decimal(emcount[words[j]]["count"])
    
    #print emprob['on']
    return emprob


def getTransitionProb(train):

    transProb={}
    transCount={}
    n = len(train)
    for i in xrange(2,n,3):
        biotags=train[i].split()
        for j in range(len(biotags)-1):
            if biotags[j][2:] in tags2:
                tag=biotags[j][2:]
            else :
                tag="O"
            if biotags[j+1][2:] in tags2:
                tag2=biotags[j+1][2:]
            else :
                tag2="O"
            if tag not in transCount:
                transCount[tag]={}
                transProb[tag]={}
            if tag2 not in transCount[tag]:
                transCount[tag][tag2]=1
                transProb[tag][tag2]=1
            else :
                transCount[tag][tag2]+=1
    
    for first in transCount:
        count=0;
        for second in transCount[first]:
            count+=transCount[first][second]
        for second in transCount[first]:
            transProb[first][second]=Decimal(transCount[first][second])/Decimal(count)
        #print transProb[first]
        #print "\n"
    return transProb
    

def viterbi(test):
    TP=getTransitionProb(alllines);
    EP=getEmissionProb(alllines);
    print EP['on']
    TP['START']={}
    TP['START']['ORG']=0.2
    TP['START']['LOC']=0.2
    TP['START']['MISC']=0.2
    TP['START']['PER']=0.2
    TP['START']['O']=0.2
    n=len(test)
    tagged=[]#[None] * n/3
    # add start transition probability - add a start tag and do this properly!
    j=0;
    
    for i in xrange(0,n,3):
        
        words=test[i].split()
        prevBestProb=1
        prevBestTag="START"
        besttagseq=[]
        for word in words:
            bestprob=0;
            besttag="O"
            for tag in tags2:
                #print TP[prevBestTag][tag]
                
                #print word
                #print EP[word]
                if word not in EP:
                    emmprob=0;
                elif  tag not in EP[word]:
                    emmprob=0
                else:
                    emmprob= EP[word][tag]
                    
                #if emmprob>0:
                    #print emmprob
                currprob=Decimal(prevBestProb)*Decimal(TP[prevBestTag][tag])*Decimal(emmprob);
                if bestprob < currprob:
                    bestprob=currprob
                    besttag=tag
                    #print besttag
            prevBestProb=bestprob
            prevBesttag=besttag
            if bestprob==0:
                prevBestProb=1
                prevBesttag='O'
            
            besttagseq.append(besttag);
        tagged.append(besttagseq)
        #print(len(tagged))
        j=j+1;
        
    createsubmission(tagged)    

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
    with open('hmmviterbi.csv','wt') as file:
        w = csv.writer(file,delimiter=" ")
        w.writerow(["Type","Prediction"])
        for key, value in finalSub.items():
            w.writerow([key]+ value)
        file.close()
    
        #break;

        
def createblocks(start,end):
    #print str(start)+"-"+str(end)
    return str(start)+"-"+str(end)
    
            
    
    
    
#print viterbi();    
        
        
    
#getEmissionProb(alllines);
viterbi(testSet);


   