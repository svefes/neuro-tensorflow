import nltk
import re
import pickle
from nltk.corpus import stopwords

def clear(path, output = 'home/sven/Desktop/output_clear'):
    '''extracts pure posts without metadata to a list and pickels it to output 
       
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

    with open(output, 'bw') as out:
        pickle.dump(res, out)

    return res

def taging(sent):
    return nltk.tokenize.word_tokenize(sent, language = 'german') #uses treebank tokenize (punctuation!!)

def valid_tagging(sent):
    stop = stopwords.words('german')
    res = tagging(sent)
    return [w for w in res if w not in stop]

def stemming(sent):
    gs = nltk.stem.snowball.GermanStemmer()
    return [gs.stem(w) for w in sent]

def get_features(sent):
    tagged = tagging(sent)
    vtagged = valid_tagging(sent)
    stemmed = stemming(vtagged)
    tc = len(tagged)
    vtc = len(vtagged)
    res = (sent, [tc, vtc, (vtc/tc), (len(set(stemmed))/vtc)])
