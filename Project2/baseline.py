import re

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
    #print lex
  		
    
    
createLexicon('./train.txt');    
