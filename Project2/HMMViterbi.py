import csv
import random
from decimal import *
from collections import defaultdict


tags=["B-LOC","B-PER","B-ORG","B-MISC", "I-LOC","I-PER","I-ORG","I-MISC","O"]
tags2=["PER","LOC","MISC","ORG"]


alllines = open('./train.txt').readlines();


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
        for second in transCount[first]:
            transProb[first][second]=Decimal(transCount[first][second])/Decimal(len(transCount[first]))
        print transProb[first]
        print "\n"
    
    
        
        
    
#getEmissionProb(alllines);
getTransitionProb(alllines);


   