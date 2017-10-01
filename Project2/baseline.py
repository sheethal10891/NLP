import re
import csv
from collections import defaultdict

def createLexicon(path):
    lines = open(path).readlines();
    n=len(lines)
    tags=["B-LOC","B-PER","B-ORG","B-MISC",	"I-LOC","I-PER","I-ORG","I-MISC"]
    lex={}
    for i in xrange(2,n,3):
        biotags=lines[i].split()
        words=lines[i-2].split()
        for j in range(len(biotags)):
            if biotags[j] in tags:
                lex[words[j]]=biotags[j][2:]
#    print lex
    return lex		
    
    
lex=createLexicon('./train.txt');

def test(lex,path):
  
    lines = open(path).readlines();
    n=len(lines)
  
    tag_dict=defaultdict(list)

    for i in xrange(0,n,3):
        tokens=lines[i].split()
	numbers=lines[i+2].split()
        token_len=len(tokens)
	for j in range(0,token_len-1):
	    if tokens[j] in lex: 
		tag=lex[tokens[j]]
		tag_dict[tag].append(int(numbers[j]))
	    else:
		tag_dict['MISC'].append(int(numbers[j]))
   
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

test(lex,'test.txt')
