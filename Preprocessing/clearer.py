import nltk
import re
import pickle
import csv
import dateutil.parser as dparser
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
    di_channel = dict()
    di_stemmed = dict()
    di_cluster = dict()
    top_channel = dict()

    #1.)sort all valid stemmed words(value) acccording to their channel(key)
    #into di_channel
    #2.)for every channel puts the most frequent 10%(rounded) of words into a list
    #
    for post in li:
        sent = post[3] #text without metadata
        vtagged = valid_tagging(sent)
        stemmed = stemming(vtagged)
        channel_id = int(post[1])
        if channel_id not in di_channel: 
            di_channel[channel_id] = stemmed
        else:
            di_channel[channel_id] = di_channel[channel_id] + stemmed
    for channel in di_channel:
        top_channel[channel] = di_channel[channel][:]
        fdist = nltk.FreqDist(top_channel[channel])
        top_channel[channel] = fdist.most_common(round(len(fdist)*0.1))
        for k, word in enumerate(top_channel[channel]):  
            top_channel[channel][k] = word[0]

    for channle in di_channel:
        di_stemmed[channel] = list()
    max_over = {channel: [0,None] for channel in di_channel}

    di_cluster = {channel: list() for channel in di_channel}
    
    li.sort(key = lambda x: x[8]) #sort by inserttime
    for i, post in enumerate(li):
        #get set of valid tagged words per post
        #get Tokencount/Validtokencount
        ch = int(post[1])
        sent = post[3] #text without metadata
        tagged = tagging(sent)
        vtagged = valid_tagging(sent)
        stemmed = stemming(vtagged)
        set_stemmed = set(stemmed)
        tc = len(tagged)
        vtc = len(vtagged)
        
        #find posting with max overlap and the distance to it for every channel
        
        for j, s in enumerate(di_stemmed[ch]):
            current_over = len(s&set_stemmed)
            if max_over[ch][0] <= current_over:
                max_over[ch][0] = current_over
                max_over[ch][1] = j+1
        if max_over[ch][1]:
            dist_over = len(di_stemmed[ch])-max_over[ch][1]+1
        else:
            dist_over = None 
        di_stemmed[ch].append(set_stemmed)

        #get Channel Overlap
        top10_channel = set(top_channel[ch])
        channel_over = len(set_stemmed&top10_channel)/len(top10_channel)

        #make clusters
        cluster_size = 0
        li_cluster = di_cluster[ch]
        max_sim = [-1, 0]
        if not li_cluster:
            li_cluster.append([post[8], set_stemmed])
        else:
            for index, cluster in enumerate(li_cluster):
                if (dparser.parse(post[8])-dparser.parse(cluster[0])).total_seconds() <= 300:
                    for st in cluster[1:]:
                        sim = len(set_stemmed&st)/min(len(st), len(set_stemmed))
                        if max_sim[1] <= sim and sim >=0.6:
                            max_sim[1] = sim
                            max_sim[0] = index
            if max_sim[0] != -1:
                li_cluster[max_sim[0]][0] = post[8]
                li_cluster[max_sim[0]].append(set_stemmed)
                cluster_size = len(li_cluster[max_sim[0]])-1
            else:
                li_cluster.append([post[8], set_stemmed])
                cluster_size = 1    

        cluster_over = len(set_stemmed&(set.union(*li_cluster[max_sim[0]][1:])))/len(set.union(*li_cluster[max_sim[0]][1:]))
        
        dur_to_prev = '0:00:00'
        if i > 0:
            dur_to_prev = dparser.parse(post[8])-dparser.parse(li[i-1][8])

        dur_to_next = '0:00:00'
        if i < len(li)-1:
            dur_to_next = dparser.parse(li[i+1][8])-dparser.parse(post[8])
	
        create_dur = dparser.parse(post[6])-dparser.parse(post[5])   
        
        dur_per_w = '0:00:00'
        if tc:
            dur_per_w = create_dur/tc 

        hold_back = dparser.parse(post[7])-dparser.parse(post[6])  

        li[i].append([tc, vtc, (vtc/tc), (len(set(stemmed))/vtc), max_over[ch][0], dist_over, 
channel_over, cluster_size, cluster_over, str(dur_to_prev), str(dur_to_next), str(create_dur), str(dur_per_w),
str(hold_back)])
   
