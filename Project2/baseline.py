import csv
import random
from decimal import *
from collections import defaultdict


tags=["B-LOC","B-PER","B-ORG","B-MISC", "I-LOC","I-PER","I-ORG","I-MISC"]

def createLexicon(train):
    n=len(train)
    print train[2042]
    print train[2040]
    lex={}
    for i in xrange(2,n,3):
        biotags=train[i].split()
        words=train[i-2].split()
        for j in range(len(biotags)):
            if biotags[j] in tags:
                if words[j]==",":
                    print biotags[j]," ",i
                lex[words[j]]=biotags[j][2:]
    return lex
    	
def calculatePrecision():
    lex=createLexicon(train)
    nVal=len(val)
    correct=0
    predicted=0
    totalTags=0
    for i in range(2,nVal,3):
        words=val[i-2].split()
        biotags=val[i].split()
        for j in range(len(words)):
            if biotags[j] in tags:
                totalTags+=1
                if words[j] in lex:
                    predicted+=1
                    tag=lex[words[j]]
                    if tag==biotags[j][2:]:
                        correct+=1
                '''
                else: # with random words!
                    predicted+=1
                    tag=random.choice(tags)
                    if tag==biotags[j]:
                        correct+=1
                        '''
    print correct
    print predicted
    print totalTags
    precision=Decimal(correct)/Decimal(predicted)
    recall=Decimal(correct)/Decimal(totalTags)
    fMeasure=2*precision*recall/(precision+recall)
    print precision
    print recall
    print fMeasure
  
    
alllines = open('./train.txt').readlines();
n=len(alllines)
train=alllines[:29400]
val=alllines[29400:] 

    
#createLexicon('./train.txt');

def test(lex,path):
  
    lines = open(path).readlines();
    n=len(lines)
  
    tag_dict=defaultdict(list)
    noun_tags=["NNPS","NNP"]
    tags=["PER","LOC","MISC","ORG"]

    for i in xrange(0,n,3):
        tokens=lines[i].split()
        pos=lines[i+1].split()
	numbers=lines[i+2].split()
        token_len=len(tokens)
	for j in range(0,token_len-1):
        if pos[j] in noun_tags:
	       if tokens[j] in lex: 
		   tag=lex[tokens[j]]
		   tag_dict[tag].append(int(numbers[j]))
	       else:
		   tag_dict[random.choice(tags)].append(int(numbers[j]))
   
    for key in tag_dict: 
        length=len(tag_dict[key])
        new_values=[]
        i=0
	while (i<length):
           start=i
	   loop=0
           
	   while(i<length-1 and tag_dict[key][i]+1 ==tag_dict[key][i+1]):
                loop=1
		i=i+1
	   if(loop==1):	
              new_values.append(str(tag_dict[key][start])+"-"+str(tag_dict[key][i]))
           else:
              new_values.append(str(tag_dict[key][start])+"-"+str(tag_dict[key][start]))
           loop=0
           i=i+1
	tag_dict[key]=new_values
#    print(tag_dict['PER'])
    
    with open('baseline.csv','wt') as file:
        w = csv.writer(file,delimiter=" ")
        w.writerow(["Type","Prediction"])
        for key, value in tag_dict.items():
           w.writerow([key]+ value)
        file.close()

#test(lex,'test.txt')
#calculatePrecision()
