import nltk
import re
import pickle
import csv
from nltk.corpus import stopwords


def pure_text(path):
    '''extracts pure posts without metadata to a list
       
        @param path: path to input
        @type path: C{str}'''
    res = list()

    reader = open(path)
    raw = reader.read()
    raw = raw[raw.find('\n')+2:] #adapt if no one line header
    raw = raw.split('\n')
    raw = [re.search('"(.*?)"', post) for post in raw]

    for p in raw:
        if p:
            res.append(p.group(1))

    return res

def listify_postings(path):
    raw = list(csv.reader(open(path)))
    raw = raw[1:] #deletes description (first line)
    return raw

def tagging(sent):
    return nltk.tokenize.word_tokenize(sent, language = 'german') #uses treebank tokenize (punctuation!!)

def valid_tagging(sent):
    stop = stopwords.words('german')
    res = tagging(sent)
    return [w for w in res if w not in stop]

def stemming(sent):
    gs = nltk.stem.snowball.GermanStemmer()
    return [gs.stem(w) for w in sent]

def get_features(li):
    li_stemmed = list()
    li.sort(key = lambda x: x[7]) #sort by submittime
    for i, post in enumerate(li):
        sent = post[3]
        tagged = tagging(sent)
        vtagged = valid_tagging(sent)
        stemmed = stemming(vtagged)
        
        max_over = [0, None]  #find posting with max overlap
        for j, s in enumerate(li_stemmed):
            current_over = len(s&set(stemmed))
            if max_over[0] <= current_over:
                max_over[0] = current_over
                max_over[1] = j+1
        
        if max_over[1]:
            dist_over = len(li_stemmed) - max_over[1] + 1
        else:
            dist_over = None        

        li_stemmed.append(set(stemmed))
        tc = len(tagged)
        vtc = len(vtagged)
        li[i].append([tc, vtc, (vtc/tc), (len(set(stemmed))/vtc), max_over[0], dist_over])
   
