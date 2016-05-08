import nltk
import re
import pickle

def clear(path, output):
    '''extracts pure posts without metadata to a list and pickels it to outp 
       
        @param path: path to input
        @type path: C{str}'''
    res = list()

    reader = open(path)
    raw = reader.read()
    raw = raw[raw.find('\n')+2:] #adapt if not one line header
    raw = raw.split('\n')
    raw = [re.search('"(.*?)"', post) for post in raw]

    for p in raw:
        if p:
            res.append(p.group(1))

    with open(output, 'bw') as out:
        pickle.dump(res, out)

    return res
