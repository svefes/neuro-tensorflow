#@TODO: AuthorIsChannelOwner is missing (no data available)
#what about normalizing
import nltk
import re
import pickle
import csv
import random
import numpy
import pymysql as sql
from datetime import datetime
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
    exp = re.compile(r'\W') #just alphanumeric strings without punctuation
    stop = stopwords.words('german') #no stop words
    res = tagging(sent)
    return [w for w in res if w not in stop and not exp.match(w)]

def stemming(sent):
    gs = nltk.stem.snowball.GermanStemmer()
    return [gs.stem(w) for w in sent]

def get_features(li):    
    di_channel = dict() #channel --> list of all valid stemmed word
    di_stemmed = dict() #channel --> list of sets of valid stemmed words per post (in order of inserttime)
    di_cluster = dict()
    top_channel = dict()
    di_time = dict() #channel --> list of all insert-timestamps
    di_author = dict() #channel --> list of author plus values

    #1.)sort all valid stemmed words(value) acccording to their channel(key)
    #into di_channel
    #2.)for every channel puts the most frequent 10%(rounded) of words into a list
    #which is saved in a dict top_channel[channel]
    for post in li:
        sent = post[3] #text without metadata
        vtagged = valid_tagging(sent)
        stemmed = stemming(vtagged)
        channel_id = int(post[1])
        if channel_id not in di_channel: 
            di_channel[channel_id] = stemmed
        else:
            di_channel[channel_id] = di_channel[channel_id] + stemmed
            
    for channel in di_channel.keys():
        top_channel[channel] = di_channel[channel][:]
        fdist = nltk.FreqDist(top_channel[channel])
        top_channel[channel] = fdist.most_common(round(len(fdist)*0.1))
        for k, word in enumerate(top_channel[channel]):  
            top_channel[channel][k] = word[0]

    #init of di_stemmed, max_over, di_cluster, di_time, punctuation-regex
    for channel in di_channel.keys():
        di_stemmed[channel] = list()
    max_over = {channel: [0,None] for channel in di_channel}
    di_cluster = {channel: list() for channel in di_channel}
    di_time = {channel: list() for channel in di_channel}
    reg_punct = re.compile(r'\W')
    reg_end = re.compile(r'[\!\?\.]')
    reg_quest = re.compile(r'\?\s*$')
    reg_break = re.compile(r'\n')
    reg_ref = re.compile(r'@(Folie|Slide)\s?[a-zA-Z]?\.?\s?(\d+)')
    reg_smile = re.compile(r'[:;BXx=][-o]?[\)/\(DPp\|]|[\^][\.,-_]?[\^]')# also matches x) which could be part of formula
    reg_script = re.compile(r'<\s*script[\s/>]')
    reg_alpha = re.compile(r'\w')


    #fill di_time, di_author
    for post in li:
        di_time[int(post[1])].append(dparser.parse(post[8]))
        if (int(post[1]),int(post[2])) in di_author:
            di_author[(int(post[1]),int(post[2]))].append(int(post[10]))
        else:
            di_author[(int(post[1]),int(post[2]))] = [int(post[10])]

    for k in di_author:
        total = len(di_author[k])
        pos = 0
        neg = 0
        for val in di_author[k]:
            if val > 0:
                pos += 1
            else:
                neg += 1
        pos_rate = pos/total
        neg_rate = neg/total
        di_author[k] = [total, pos_rate, neg_rate]

    #sort values of di_time according to inserttime
    for l in di_time.values():
        l.sort()
        l.insert(0, l[0])
        l.append(l[-1])
        mean = sum((l[i]-l[i-1]).total_seconds() for i in range(2,len(l)-2))/len(l)-3
        l.append(mean)
    
    li.sort(key = lambda x: x[8]) #sort by inserttime
    #iterates over all posts and extracts their features
    for i, post in enumerate(li):
        #get set of valid tagged words per post
        #get Tokencount/Validtokencount
        ch = int(post[1])
        au = int(post[2])
        sent = post[3] #text without metadata
        tagged = tagging(sent)
        vtagged = valid_tagging(sent)
        stemmed = stemming(vtagged)
        set_stemmed = set(stemmed)
        tc = len(tagged) #normalizing with min/max?
        vtc = len(vtagged) #normalizing with min/max?
        non_stop = 0
        dist_ratio = 1
        if tc > 0:
            non_stop = vtc/tc
        if vtc > 0:
            dist_ratio = (len(set(stemmed))/vtc)

        #find posting with max overlap and the distance to it
        for j, s in enumerate(di_stemmed[ch]):
            current_over = len(s&set_stemmed)
            if max_over[ch][0] <= current_over:
                max_over[ch][0] = current_over
                max_over[ch][1] = j+1
        if max_over[ch][1]:
            dist_over = len(di_stemmed[ch])-max_over[ch][1]+1 #  normalizing with total number of posting
        else:
            dist_over = 0 
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
                        sim = 0 
                        if len(set_stemmed) > 0 and len(st) > 0:
                            sim = len(set_stemmed&st)/min(len(st), len(set_stemmed))
                        if max_sim[1] <= sim and sim >=0.6:
                            max_sim[1] = sim
                            max_sim[0] = index
            if max_sim[0] != -1:
                li_cluster[max_sim[0]][0] = post[8]
                li_cluster[max_sim[0]].append(set_stemmed)
                cluster_size = len(li_cluster[max_sim[0]])-1 #normalizing with min/max cluster size
            else:
                li_cluster.append([post[8], set_stemmed])
                cluster_size = 1    

        cluster_over = 0
        if len(set.union(*li_cluster[max_sim[0]][1:])) > 0:
            cluster_over = len(set_stemmed&(set.union(*li_cluster[max_sim[0]][1:])))/\
len(set.union(*li_cluster[max_sim[0]][1:]))

        #calculate time features 
        current_time = di_time[ch][1]
        prev_time = di_time[ch][0]
        next_time = di_time[ch][2]
            
        
        dur_to_prev = (current_time - prev_time).total_seconds() #normalizing with min/max

        comp_freq = dur_to_prev/di_time[ch][-1]
            
        dur_to_next = (next_time - current_time).total_seconds() #normalizing with min/max
        del di_time[ch][0]
        
        create_dur = (dparser.parse(post[6])-dparser.parse(post[5])).total_seconds()  #normalizing with min/max 
        
        dur_per_w = 0
        if tc:
            dur_per_w = create_dur/tc 

        hold_back = (dparser.parse(post[7])-dparser.parse(post[6])).total_seconds() #normalizing with min/max
        
        #get content features
        post_type =1
        if post[9] == 'IMG':
            post_type = 2
        elif post[9] == 'EQU':
            post_type = 3

        is_handmade = 0
        if post_type == 2:
            is_handmade = post[11]
        
        punct = 0
        if reg_punct.search(sent):
            punct += 0.5
        if reg_end.search(sent):
            punct += 0.5

        ends_with_quest = 0
        if reg_quest.search(sent):
            ends_with_quest = 1

        line_break = 0
        if reg_break.search(sent):
            line_break = 1

        smiley = 0
        if reg_smile.search(sent):
            smiley = 1

        alpha = 0
        if reg_alpha.match(sent):
            alpha = 1
            
        script_tag = 0
        if reg_script.search(sent):
            script_tag = 1

        #get author features !! AuthorIsChannelOwner is missing (no data available)
        post_ratio = di_author[(ch,au)][0]
        
        pos_ratio = di_author[(ch,au)][1]
        
        neg_ratio = di_author[(ch,au)][2]
        
        li[i].append([tc, vtc, non_stop, dist_ratio, max_over[ch][0], dist_over, 
channel_over, cluster_size, cluster_over, dur_to_prev, comp_freq, dur_to_next,
create_dur, dur_per_w, hold_back, post_type, is_handmade, punct, ends_with_quest,
line_break, smiley, alpha, script_tag, post_ratio, pos_ratio, neg_ratio])

def train_test(l , rate = 6/7): # 6/7 is the rate of the MNIST data
    random.seed(datetime.now())
    l_copy = l[:]
    random.shuffle(l_copy)
    return (l_copy[0:round(rate*len(l_copy))], l_copy[round(rate*len(l_copy)):]) 
    
def next_batch(l, n = 100):
    random.seed(datetime.now())
    if n=='all':
        rand = [random.choice(l) for i in range(len(l))]
    else:
        rand = [random.choice(l) for i in range(n)]
    features = list()
    assess = list()
    for li in rand:
        features.append(li[-1])
        assess.append(int(li[-2])+1)
    assess = numpy.eye(3)[assess] # make one-hot-vector for every input in assess
    return (features, assess)

def init():
#delete after testing
    li = listify_postings('Doepke.csv')
    get_features(li)
    train, test = train_test(li)
    return (li, train, test)
