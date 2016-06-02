#@TODO: AuthorIsChannelOwner is missing (no data available)
#what about normalizing
#prevent double assessment of same posts
#needs database "wescript" for 'sven'@localhost with passwrd='nevs'
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
    #connect to db "wecsript" on localhost
    con = sql.connect (host = 'localhost', user = 'sven', passwd = 'nevs', db='wescript')
    cur = con.cursor()
    cur.execute('CREATE TABLE IF NOT EXISTS assessedPosts (post_id INT PRIMARY KEY, ch_id INT,\
tokenCount DOUBLE, n_tokenCount DOUBLE, validTokenCount DOUBLE, n_validTokenCount DOUBLE, nonStopwordRatio DOUBLE,\
distinctWordRatio DOUBLE, overlapPrevious DOUBLE, overlapDistance DOUBLE, n_overlapDistance DOUBLE,\
overlapChannel DOUBLE, clusterSize DOUBLE, n_clusterSize DOUBLE, overlapCluster DOUBLE, durationToPrevious DOUBLE,\
n_durationToPrevious DOUBLE, comparedFrequency DOUBLE, durationToNext DOUBLE, n_durationToNext DOUBLE,\
creationDuration DOUBLE, n_creationDuration DOUBLE, durationPerWord DOUBLE, n_durationPerWord DOUBLE,\
holdBackDuration DOUBLE, n_holdBackDuration DOUBLE, postingTypeText BOOLEAN, postingTypeImage BOOLEAN,\
postingTypeEquation BOOLEAN, imageIsHandmade BOOLEAN, punctuationLevel FLOAT,\
endsWithQuestionmark BOOLEAN, containsLinebreak BOOLEAN, containsSlideReference BOOLEAN,\
containsSmiley BOOLEAN, beginsWithAlphanumeric BOOLEAN, containsScriptTags BOOLEAN,\
authorPostingRatio DOUBLE, authorPositiveRatio DOUBLE, authorNegativeRatio DOUBLE,\
authorIsChannelOwner BOOLEAN)')
    cur.execute('CREATE TABLE IF NOT EXISTS channelStatistics (ch_id INT PRIMARY KEY,\
tokenCountMin DOUBLE, tokenCountMax DOUBLE, valdiTokenCountMin DOUBLE, validTokenCountMax DOUBLE,\
overlapDistanceMin DOUBLE,overlapDistanceMax DOUBLE, clusterSizeMin DOUBLE, clusterSizeMax DOUBLE,\
durationToPreviousMin DOUBLE, durationToPreviousMax DOUBLE,durationToNextMin DOUBLE, durationToNextMax DOUBLE,\
creationDurationMin DOUBLE, creationDurationMax DOUBLE, holdBackDurationMin DOUBLE,\
holdBackDurationMax DOUBLE, stemmedPosts BLOB, stemmedList BLOB, authorDict BLOB, clusterDict BLOB)')

    di_max = dict() 
    di_min = dict()
    
    di_channel = dict() #channel --> list of all valid stemmed word
    di_stemmed = dict() #channel --> set of lists of valid stemmed words per post (in order of inserttime)
    di_cluster = dict()
    top_channel = dict()
    di_author = dict() #channel --> list of author plus values
    cur.execute('SELECT post_id FROM assessedPosts')
    already_assessed = [l[0] for l in list(cur)]
    li = [x for x in li if int(x[0]) not in already_assessed]
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
        if channel_id not in di_author:
            di_author[channel_id] = {int(post[2]):[int(post[10])]}
        elif int(post[2]) not in di_author[channel_id]:
            di_author[channel_id].update({int(post[2]):[int(post[10])]})
        else:
            di_author[channel_id][int(post[2])].append(int(post[10]))
    print(di_author)
    for channel in di_channel.keys():
        #total author values
        for au in di_author[channel]:
            total = len(di_author[channel][au])
            pos = 0
            neg = 0
            for val in di_author[channel][au]:
                if val > 0:
                    pos += 1
                else:
                    neg += 1
            di_author[channel][au] = [total, pos, neg]
        #add lists from db
        if cur.execute('SELECT stemmedList, authorDict FROM channelStatistics WHERE ch_id=%s', (channel,)):
            fetch = cur.fetchone()
            di_channel[channel] = di_channel[channel] + pickle.loads(fetch[0])
            temp_author = pickle.loads(fetch[1])
            for au in temp_author:
                if au not in di_author[channel]:
                    di_author[channel].update({au:temp_author[au]})
                else:
                    di_author[channel][au][0] += temp_author[au][0]
                    di_author[channel][au][0] += temp_author[au][1]
                    di_author[channel][au][0] += temp_author[au][2]
    print(di_author)           
    #init of di_stemmed, di_cluster, punctuation-regex
    for channel in di_channel.keys():
        if cur.execute('SELECT stemmedPosts, clusterDict FROM channelStatistics WHERE ch_id=%s', (channel,)):
            fetch = cur.fetchone()
            di_stemmed[channel] = pickle.loads(fetch[0])
            di_cluster[channel] = pickle.loads(fetch[1])
        else:
            di_stemmed[channel] = list()
            di_cluster[channel] = list()
            
    reg_punct = re.compile(r'\W')
    reg_end = re.compile(r'[\!\?\.]')
    reg_quest = re.compile(r'\?\s*$')
    reg_break = re.compile(r'\n')
    reg_ref = re.compile(r'@(Folie|Slide)\s?[a-zA-Z]?\.?\s?(\d+)')
    reg_smile = re.compile(r'[:;BXx=][-o]?[\)/\(DPp\|]|[\^][\.,-_]?[\^]')# also matches x) which could be part of formula
    reg_script = re.compile(r'<\s*script[\s/>]')
    reg_alpha = re.compile(r'\w')

    li.sort(key = lambda x: x[8]) #sort by inserttime
    #iterates over all posts and extracts their features
    for i, post in enumerate(li):
        #get set of valid tagged words per post
        #get Tokencount/Validtokencount
        p_id = int(post[0]) 
        ch = int(post[1])
        au = int(post[2])
        time_inter = 0 #the current time key is somewhere between old time keys
        current_time = dparser.parse(post[8])
        sent = post[3] #text without metadata
        tagged = tagging(sent)
        vtagged = valid_tagging(sent)
        stemmed = stemming(vtagged)
        set_stemmed = set(stemmed)
        tc = len(tagged) 
        vtc = len(vtagged)

        #tc = (tc - di_min_tc[ch])/di_max_tc[ch]
        #vtc = (tc - di_min_vtc[ch])/di_max_vtc[ch]
        
        non_stop = 0
        dist_ratio = 1
        if tc > 0:
            non_stop = vtc/tc
        if vtc > 0:
            dist_ratio = (len(set(stemmed))/vtc)
   
        #insert stemmed set with timestamp at correct position
        fin = len(di_stemmed[ch])  
        if di_stemmed[ch] and current_time < di_stemmed[ch][-1][1]:
            for j, s in enumerate(di_stemmed[ch]):
                if current_time < s[1]:
                    di_stemmed[ch].insert(j,[p_id, current_time, set_stemmed, au])
                    fin = j
                    break
        else:
            di_stemmed[ch].append([p_id, current_time, set_stemmed, au])
        #find posting with max overlap and the distance to it
        max_over = [0, None]
        for j, s in enumerate(di_stemmed[ch][:fin]):
            if len(s[2])>0:
                current_over = len(s[2]&set_stemmed)/len(s[2])
            else:
                current_over = 0
            if max_over[0] <= current_over:
                max_over[0] = current_over
                max_over[1] = j
        if max_over[1]:
            dist_over = fin-max_over[1]
        else:
            dist_over = 0
        #if set is inserted somewhere inbetween, compute overlap again for all following
        if fin < len(di_stemmed[ch])-1:
            time_inter = 1
            for j, s in enumerate(di_stemmed[ch][fin+1:]):
                max_over = [0, None]
                for k, t in enumerate(di_stemmed[ch][:fin+1+j]):
                    if len(t[2]):
                        current_over = len(t[2]&s[2])/len(t[2])
                    else:
                        current_over = 0
                    if max_over[0] <= current_over:
                        max_over[0] = current_over
                        max_over[1] = k
                cur.execute('UPDATE assessedPosts SET overlapPrevious=%s, overlapDistance=%s WHERE post_id=%s'\
, (max_over[0], fin+1+j-max_over[1], s[0]))
                    
        #make clusters
        cluster_over = 1
        cluster_size = 1
        #if current_time is somewhere in the middle, all posts possibly have to be assigned to new clusters
        if time_inter:
            di_cluster[ch] = list()
            for j,s in enumerate(di_stemmed[ch]):
                cluster_size = 1
                li_cluster = di_cluster[ch]
                max_sim = [-1, 0]
                if not li_cluster:
                    li_cluster.append([s[1],s[2]])
                else:
                    for index, cluster in enumerate(li_cluster):
                        if (s[1]-cluster[0]).total_seconds() <= 300:
                            for st in cluster[1:]:
                                sim = 0 
                                if len(s[2]) > 0 and len(st) > 0:
                                    sim = len(s[2]&st)/min(len(st), len(s[2]))
                                if max_sim[1] <= sim and sim >=0.6:
                                    max_sim[1] = sim
                                    max_sim[0] = index
                    if max_sim[0] != -1:
                        li_cluster[max_sim[0]][0] = s[1]
                        li_cluster[max_sim[0]].append(s[2])
                        cluster_size = len(li_cluster[max_sim[0]])-1 #normalizing with min/max cluster size
                    else:
                        li_cluster.append([s[1], s[2]])
                        cluster_size = 1    

                cluster_over = 0
                if len(set.union(*li_cluster[max_sim[0]][1:])) > 0:
                    cluster_over = len(s[2]&(set.union(*li_cluster[max_sim[0]][1:])))/\
len(set.union(*li_cluster[max_sim[0]][1:]))
                    
                if cur.execute('SELECT * FROM assessedPosts WHERE post_id=%s',(s[0],)):
                    cur.execute('UPDATE assessedPosts SET clusterSize=%s, overlapCluster=%s WHERE post_id=%s', (cluster_size, cluster_over,s[0]))
                else:
                    cur.execute('INSERT INTO assessedPosts (clusterSize, overlapCluster) VALUES (%s, %s)', (cluster_size, cluster_over))
        #no time break
        else:                            
            li_cluster = di_cluster[ch]
            max_sim = [-1, 0]
            if not li_cluster:
                li_cluster.append([current_time, set_stemmed])
            else:
                for index, cluster in enumerate(li_cluster):
                    if (current_time-cluster[0]).total_seconds() <= 300:
                        for st in cluster[1:]:
                            sim = 0 
                            if len(set_stemmed) > 0 and len(st) > 0:
                                sim = len(set_stemmed&st)/min(len(st), len(set_stemmed))
                            if max_sim[1] <= sim and sim >=0.6:
                                max_sim[1] = sim
                                max_sim[0] = index
                if max_sim[0] != -1:
                    li_cluster[max_sim[0]][0] = current_time
                    li_cluster[max_sim[0]].append(set_stemmed)
                    cluster_size = len(li_cluster[max_sim[0]])-1 #normalizing with min/max cluster size
                else:
                    li_cluster.append([current_time, set_stemmed])
                    cluster_size = 1    

            
            if max_sim[0] != -1 and len(set.union(*li_cluster[max_sim[0]][1:])) > 0:
                cluster_over = len(set_stemmed&(set.union(*li_cluster[max_sim[0]][1:])))/\
len(set.union(*li_cluster[max_sim[0]][1:]))
                       

        #calculate time features        
        create_dur = (dparser.parse(post[6])-dparser.parse(post[5])).total_seconds()  #normalizing with min/max 
        
        dur_per_w = 0
        if tc:
            dur_per_w = create_dur/tc 

        hold_back = (dparser.parse(post[7])-dparser.parse(post[6])).total_seconds() #normalizing with min/max
        
        #get content features
        post_type_t = 1
        post_type_i = 0
        post_type_e = 0
        if post[9] == 'IMG':
            post_type_i = 1
            post_type_t = 0
        elif post[9] == 'EQU':
            post_type_e = 1
            post_type_t = 0

        is_handmade = 0
        if post_type_i == 1:
            is_handmade = int(post[11])
        
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

        slide_ref = 0
        if reg_ref.search(sent):
            slide_ref = 1

        smiley = 0
        if reg_smile.search(sent):
            smiley = 1

        alpha = 0
        if reg_alpha.match(sent):
            alpha = 1
            
        script_tag = 0
        if reg_script.search(sent):
            script_tag = 1

        #search for max/min in current session or in database
        if ch in di_max:
            if di_max[ch][0] < tc:
                di_max[ch][0] = tc
            if di_min[ch][0] > tc:
                di_min[ch][0] = tc
            if di_max[ch][1] < vtc:
                di_max[ch][1] = vtc
            if di_min[ch][1] > vtc:
                di_min[ch][1] = vtc
            if di_max[ch][2] < dist_over:
                di_max[ch][2] = dist_over
            if di_min[ch][2] > dist_over:
                di_min[ch][2] = dist_over
            if di_max[ch][3] < cluster_size:
                di_max[ch][3] = cluster_size
            if di_min[ch][3] > cluster_size:
                di_min[ch][3] = cluster_size
            if di_max[ch][6] < create_dur:
                di_max[ch][6] = create_dur
            if di_min[ch][6] > create_dur:
                di_min[ch][6] = create_dur
            if di_max[ch][7] < dur_per_w:
                di_max[ch][7] = dur_per_w
            if di_min[ch][7] > dur_per_w:
                di_min[ch][7] = dur_per_w
            if di_max[ch][8] < hold_back:
                di_max[ch][8] = hold_back
            if di_min[ch][8] > hold_back:
                di_min[ch][8] = hold_back
                
        else:
            if cur.execute('SELECT * FROM channelStatistics WHERE ch_id=%s', (ch,)):
                fetch = cur.fetchone()
                di_min[ch][0] = fetch[0]
                di_max[ch][0] = fetch[1]
                di_min[ch][1] = fetch[2]
                di_max[ch][1] = fetch[3]
                di_min[ch][2] = fetch[4]
                di_max[ch][2] = fetch[5]
                di_min[ch][3] = fetch[6]
                di_max[ch][3] = fetch[7]
                di_min[ch][4] = fetch[8]
                di_max[ch][4] = fetch[9]
                di_min[ch][5] = fetch[10]
                di_max[ch][5] = fetch[11]
                di_min[ch][6] = fetch[12]
                di_max[ch][6] = fetch[13]
                di_min[ch][7] = fetch[14]
                di_max[ch][7] = fetch[15]
                di_min[ch][8] = fetch[16]
                di_max[ch][8] = fetch[17]
            else:
                di_max[ch]=[tc, vtc,dist_over, cluster_size,0,0,create_dur,dur_per_w, hold_back]
                di_min[ch]=[tc, vtc,dist_over, cluster_size,0,0,create_dur,dur_per_w, hold_back]#0 for dur to prev can be used as first one will always have 0
                

        if cur.execute('SELECT * FROM assessedPosts WHERE post_id = %s', p_id):
            cur.execute('UPDATE assessedPosts SET tokenCount=%s, validTokenCount=%s,nonStopwordRatio=%s,\
distinctWordRatio=%s, overlapPrevious=%s, overlapDistance=%s,\
creationDuration=%s, durationPerWord=%s,\
holdBackDuration=%s, postingTypeText=%s, postingTypeImage=%s,\
postingTypeEquation=%s, imageIsHandmade=%s, punctuationLevel=%s,\
endsWithQuestionmark=%s, containsLinebreak=%s, containsSlideReference=%s,\
containsSmiley=%s, beginsWithAlphanumeric=%s, containsScriptTags=%s WHERE post_id=%s',\
(tc, vtc, non_stop, dist_ratio, max_over[0], dist_over,create_dur, dur_per_w, hold_back, post_type_t,\
post_type_i, post_type_e, is_handmade, punct, ends_with_quest,line_break, slide_ref, smiley, alpha, script_tag, p_id))
        else:
            cur.execute('INSERT INTO assessedPosts VALUES\
(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)',\
(p_id,ch, tc, 0, vtc, 0, non_stop, dist_ratio, max_over[0], dist_over,0,0, cluster_size, 0, cluster_over,0, 0, 0, 0, 0,\
create_dur, 0, dur_per_w, 0, hold_back, 0, post_type_t, post_type_i, post_type_e,is_handmade, punct, ends_with_quest,\
line_break, slide_ref, smiley, alpha, script_tag,0,0,0,0))            
        con.commit()

    #calculate featues for which overall knowledge is needed  
    for ch in di_channel:
        #compute top 10% of the channel    
        top_channel = di_channel[ch][:]
        fdist = nltk.FreqDist(top_channel)
        top_channel = fdist.most_common(round(len(fdist)*0.1))
        for k, word in enumerate(top_channel):  
            top_channel[k] = word[0]
        top10_channel = set(top_channel)
        #mean time between posts in a channel
        mean = sum((di_stemmed[ch][i][1]-di_stemmed[ch][i-1][1]).total_seconds()\
for i in range(1,len(di_stemmed[ch])-1))/(len(di_stemmed[ch])-1)
        
        for j, st in enumerate(di_stemmed[ch]):
            au = st[3]
            channel_over = len(st[2]&top10_channel)/len(top10_channel)
            
            comp_freq = len(st[2])/mean
            dur_to_prev = 0
            if j > 0:
                dur_to_prev = (di_stemmed[ch][j][1] - di_stemmed[ch][j-1][1]).total_seconds()
            dur_to_next = 0   
            if j < len(di_stemmed[ch])-1:
                dur_to_next = (di_stemmed[ch][j+1][1] - di_stemmed[ch][j][1]).total_seconds()

            if di_max[ch][4] < dur_to_prev:
                di_max[ch][4] = dur_to_prev
            if di_min[ch][4] > dur_to_prev:
                di_min[ch][4] = dur_to_prev
            if di_max[ch][5] < dur_to_next:
                di_max[ch][5] = dur_to_next
            if di_min[ch][5] > dur_to_next:
                di_min[ch][5] = dur_to_next 

            post_ratio = di_author[ch][au][0]/len(di_stemmed[ch])
        
            pos_ratio = di_author[ch][au][1]/di_author[ch][au][0]
        
            neg_ratio = di_author[ch][au][2]/di_author[ch][au][0]

            cur.execute('UPDATE assessedPosts SET overlapChannel=%s, comparedFrequency=%s, durationToPrevious=%s,\
durationToNext=%s, authorPostingRatio=%s, authorPositiveRatio=%s, authorNegativeRatio=%s WHERE post_id=%s', (channel_over, comp_freq,\
dur_to_prev, dur_to_next, post_ratio, pos_ratio, neg_ratio, di_stemmed[ch][j][0]))
        con.commit()
        #normalize features of ALL posts in a channel
        cur.execute('SELECT * FROM assessedPosts WHERE ch_id=%s', (ch,))
        for row in cur:
            n_tc = n_vtc = n_dist_over = n_cluster_size = n_dur_to_prev = n_dur_to_next = n_create_dur = n_dur_per_w = n_hold_back = 0
            if di_max[ch][0]>0:
                n_tc = (row[2] - di_min[ch][0])/di_max[ch][0]
            if di_max[ch][1]>0:
                n_vtc = (row[4] - di_min[ch][1])/di_max[ch][1]
            if di_max[ch][2]>0:
                n_dist_over = (row[10] - di_min[ch][2])/di_max[ch][2]
            if di_max[ch][3]>0:
                print('debug')
                n_cluster_size = (row[13] - di_min[ch][3])/di_max[ch][3]
            if di_max[ch][4]>0:
                n_dur_to_prev  = (row[16] - di_min[ch][4])/di_max[ch][4]
            if di_max[ch][5]>0:
                n_dur_to_next = (row[18] - di_min[ch][5])/di_max[ch][5]
            if di_max[ch][6]>0:
                n_create_dur = (row[20] - di_min[ch][6])/di_max[ch][6]
            if di_max[ch][7]>0:
                n_dur_per_w = (row[22] - di_min[ch][7])/di_max[ch][7]
            if di_max[ch][8]>0:
                n_hold_back = (row[24] - di_min[ch][8])/di_max[ch][8]
            cur.execute('UPDATE assessedPosts SET n_tokenCount=%s, n_validTokenCount=%s, n_overlapDistance=%s,\
n_clusterSize=%s, n_durationToPrevious=%s, n_durationToNext=%s, n_creationDuration=%s,n_durationPerWord=%s, n_holdBackDuration=%s WHERE post_id=%s',\
(n_tc, n_vtc, n_dist_over, n_cluster_size, n_dur_to_prev, n_dur_to_next, n_create_dur, n_dur_per_w, n_hold_back, row[0]))
        con.commit()

        #save all statistics for later use   
        if cur.execute('SELECT * FROM channelStatistics WHERE ch_id=%s', ch):
            cur.execute('UPDATE channelStatistics SET tokenCountMin=%s, tokenCountMax=%s, valdiTokenCountMin=%s, validTokenCountMax=%s,\
overlapDistanceMin=%s ,overlapDistanceMax=%s , clusterSizeMin=%s , clusterSizeMax=%s ,\
durationToPreviousMin=%s, durationToPreviousMax=%s ,durationToNextMin=%s , durationToNextMax=%s,\
creationDurationMin=%s , creationDurationMax=%s, holdBackDurationMin=%s ,\
holdBackDurationMax=%s, stemmedPosts=%s, stemmedList=%s, authorDict=%s clusterDict=%s WHERE ch_id=%s',\
(di_min[ch][0], di_max[ch][0],\
di_min[ch][1], di_max[ch][1], di_min[ch][2], di_max[ch][2],\
di_min[ch][3],di_max[ch][3], di_min[ch][4], di_max[ch][4],\
di_min[ch][5],di_max[ch][5],di_min[ch][6],di_max[ch][6],\
di_min[ch][7],di_max[ch][7],\
pickle.dumps(di_stemmed[ch]), pickle.dumps(di_channel[ch]),pickle.dumps(di_author[ch]),\
pickle.dumps(di_cluster[ch]), ch))
        else:
            cur.execute('INSERT INTO channelStatistics VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)',\
(ch,di_min[ch][0], di_max[ch][0],\
di_min[ch][1], di_max[ch][1], di_min[ch][2], di_max[ch][2],\
di_min[ch][3],di_max[ch][3], di_min[ch][4], di_max[ch][4],\
di_min[ch][5],di_max[ch][5],di_min[ch][6],di_max[ch][6],\
di_min[ch][7],di_max[ch][7],\
pickle.dumps(di_stemmed[ch]), pickle.dumps(di_channel[ch]),pickle.dumps(di_author[ch]),\
pickle.dumps(di_cluster[ch])))
        con.commit()

    con.close()

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
