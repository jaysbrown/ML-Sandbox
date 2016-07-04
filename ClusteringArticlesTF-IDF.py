# coding: utf-8

# imports
import requests
import urllib.request
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from collections import defaultdict
from string import punctuation
from heapq import nlargest
from math import log
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import MeanShift
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
#from sklearn.metrics import pairwise_distances
import pandas as pd
import seaborn as sns
import re
import matplotlib.pyplot as plt
import numpy as np

sns.set_style(style='whitegrid')
sns.set_context('notebook')

def getDoxyDonkeyText(testUrl,token='post-body'):
    '''## define function to get the text,title from an individual post on DoxyDonkey Blog'''
    response = requests.get(testUrl)
    soup = BeautifulSoup(response.content,'lxml')
    page = str(soup)
    title = soup.find('title').text
    mydivs = soup.findAll('div', {'class':token})
    text = ''.join(map(lambda p: p.text,mydivs))
    return text,title

def getAllDoxyDonkeyPosts(url,links):
    '''## function to recursively get all links from DoxyDonkey tech blog'''
    response = requests.get(url)
    soup = BeautifulSoup(response.content,'lxml')
    
    for a in soup.findAll('a'):
        try:
            url = a['href']
            title = a['title']
            if title == 'Older Posts':
                print(title,url)
                links.append(url)
                getAllDoxyDonkeyPosts(url,links)
        except:
            title = ''
    return

class FrequencySummarizer:
    '''# Feature Extraction!
    most important words, most important sentences'''
    def __init__(self, min_cut=0.1, max_cut=0.9):
        # __init__ is called the 'Constructor'
        # The constructor is called every time an object is created (instantiated)
        # The special keyword 'self' is passed in as the first arg
        # of each method
        # member variables are assigned to the input values {min,max}_cut
        self._min_cut = min_cut
        self._max_cut = max_cut
        self._stopwords = set(stopwords.words('english')+list(punctuation)
                              +[r"'s",r"’s",'"','said','would','one','like','new','also',"''","'",'get','mr.','—',
                                 '”','``','could',"it’s","’",'don’t',u'million',u'billion',u'last','according',
                                'companies','service','services','business','year','first'])
        
    
    # For Variables defined here, outside a member function but inside a class,
    # then the variable becomes STATIC, this means that the variable belongs to
    # EACH instance of the class, and not to any individual instance (object)
    # of the class
    def _compute_frequencies(self,word_sent,customStopWords=None):
        # this member function (method) takes in the 'self' object as its 1st parameter
        # this method takes in a list of sentences (word_sent) and outputs a dictionary (freq)
        # in which the keys are words, and the values are the frequencies
        # of those words in the article (set of sentences)
        freq = defaultdict(int)
        # defaultdict, inherits from dictionary, but takes in a function in its constructor,
        # 'int' in this case, and will create an object of that type if the defaultdict (freq),
        # is called with a key that is not in the defaultdict (freq)
        
        if customStopWords is None:
            stopwords = set(self._stopwords)
        else:
            stopwords = set(customStopWords).union(self._stopwords)
        #tokenize article into sentences
        for sentence in word_sent:
            #tokenize each sentence into words
            for word in sentence:
                # if word is not a stopword
                if word not in self._stopwords:
                    # performs a simple counting of words in the document, neat!
                    # if a word(key) is not already in the defaultdict(freq), then
                    # it gets added with a value of type int, for the argument 'int' starting at zero
                    freq[word] += 1
        
        # Normalize the frequencies by dividing by the highest
        # filter out high/low frequncies
        # 'almost stopwords' upper limit cutoff
        # 'very rare words' lower limit cutoff
        # find the highest frequency value
        max_freq = float(max(freq.values()))
        
        for word in freq.keys():
            # overwrite the absolute frequency values with a normalized frequency 
            freq[word] = freq[word]/max_freq
            # check for normalized frequencies that are outside of the cutoffs
            if freq[word] >= self._max_cut or freq[word] <= self._min_cut:
                # delete those entries
                # del freq[word] 
                '''had to modify this because Python 3 was throwing an error for 
                dictionary changing size during the loop'''
                freq[word]=0
        return freq
        # done with this compute frequncies method!
    
    def extractFeatures(self,article,n,customStopWords=None):
        '''takes in article tuple with text as first item, title as second item,
        n is the number of features to extract, set n<0 to extract all words as features
        returns the "n" most significant words in the article
        '''
        text = article[0]
        title = article[1]
        sentences = sent_tokenize(text)
        # list of sentences, tokenized into words
        word_sent = [word_tokenize(s.lower()) for s in sentences]
        # call _compute_frequencies with word_sent
        self._freq = self._compute_frequencies(word_sent,customStopWords)
        if n < 0:
            # if called with n <0 return all words in ranked order
            return nlargest(len(self._freq.keys()),self._freq,key=self._freq.get)
        else:
            return nlargest(n,self._freq,key=self._freq.get)
    
    def extractRawFrequencies(self,article):
        '''this method returns the 'raw' word frequencies in an article'''
        text = article[0]
        title = article[1]
        # tokenize article into sentences
        sentences = sent_tokenize(text)
        # tokenize sentences into words
        # store this in a list of sentences that are lists of tokenized words
        word_sent = [word_tokenize(s.lower()) for s in sentences]
        freq = defaultdict(int)
        for s in word_sent:
            for w in s:
                if w not in self._stopwords:
                    freq[w]+=1
        return freq
            
    
    def summarize(self, article, n):
        '''return a list of the n most significant sentences from an article
            article is tuple (text,title)
            n is the number of most significant sentences to return'''
        text = article[0]
        title = article[1]
        sentences = sent_tokenize(text)
        assert n <= len(sentences)
        word_sent = [word_tokenize(s.lower()) for s in sentences]
        # calculate the freq defaultdict using the method _compute_frequencies
        self._freq = self._compute_frequencies(word_sent) 
        # setup ranking defaultdict to hold the rankings of the sentences
        ranking = defaultdict(int)
        # iterate through each sentence
        for i,sent in enumerate(word_sent):
            # iterate through each word
            for word in sent:
                if word in self._freq:
                    # add each word's normalized frequency to the ranking for the sentence 'i'
                    ranking[i] += self._freq[word]
        # this way, passing the method "get" on ranking, sorts based on Values
        # returns Keys in order of ranked values, very cool
        sents_idx = nlargest(n,ranking, key=ranking.get)
        return [sentences[j] for j in sents_idx]


def GetNewsArticles():
    ''''''
    pass

def getWashPostText(url,token):
    '''this function takes in the URL of an article in the WashPo
    returns the text of the article minus all of the junk
    '''
    try:
        '''# this works but intermittently, getting forbidden 403 from washpo numerous trials
        page = urllib.request.urlopen(url).read().decode('utf8')'''
        # alternative that works more reliably
        response = requests.get(url)
    # instantiate soup object
    
    except:
        # if unable to download the function exits early! Returning (None,None)
        return (None,None)
    
    # initialize a Beautiful Soup Object with the page we downloaded
    '''# this works but intermittently, getting forbidden 403 from washpo numerous trials
    soup = BeautifulSoup(page,'lxml')'''
    # alternative
    soup = BeautifulSoup(response.content,'lxml')
    # handle the case where the above download failed
    if soup is None:
        # if unable to parse, exits the function early returns None,None
        return (None,None)
    
    # if we get to here, it means the error checks were successful,
    # were able to parse the page
    
    text = ''
    # first search the page for the demarcation tags specified in 'token'
    # tags: <'token'> ***stuff*** </'token'>
    if soup.find_all(token) is not None:
        text = ' '.join(map(lambda p: p.text, soup.find_all(token)))
        # need to repeat this procedure to get rid of the paragraph tags and retain text <p> ***text*** </p>
        soup2 = BeautifulSoup(text,'lxml')
        if soup2.find_all('p') is not None:
            text = ' '.join(map(lambda p: p.text, soup2.find_all('p')))
    return text, soup.title.text

def getNYTText(url,token):
    '''# NYT text extractor'''
    # alternative to urllib request
    response = requests.get(url)
    # instantiate soup object
    soup = BeautifulSoup(response.content,'lxml')
    page = str(soup)
    title = soup.find('title').text
    mydivs = soup.findAll('p' , {'class': 'story-body-text story-content'})
    text = ''.join(map(lambda p: p.text, mydivs))
    return text,title

def scrapeSource(url, magicFrag='2016', scraperFunction=getNYTText, token='None'):
    ''' # Function that takes in the URL of a section of a newspaper 
        # and finds all the URLs for articles in that section, excluding
        # non-news "articles" such as advertisements
        # "Typically" news articles include a date in the url, will use that to identify
        # News articles and drop everything that doesn't have a date in the url'''
    urlBodies = {}
    '''
    had lots of 403 forbidden errors with urllib on WashPo, only succeeded occasionally
    request = urllib.request.Request(url)
    response = urllib.request.urlopen(request).read()'''
    # alternative seems more reliable for washpo
    response = requests.get(url)
    '''soup = BeautifulSoup(response,'lxml')'''
    # alternative seems more reliable for washpo
    soup = BeautifulSoup(response.content,'lxml')
    # we have a soup object, going to sort through it for the links to other articles on the page
    # links are of the form <a href='link-url'> link-text </a>
    # soup.findAll returns an iterable bs4.element.ResultSet
    # each iterable 'a' acts sorta like a dictionary
    # tag[key] returns the value of the 'key' attribute for the tag, and throws an exception if it's not there.
    
    print('\nscrapeSource Initialization')
    print('url={}'.format(url))
    print('Number of Unique Links Found={}'.format(len(set(soup.findAll('a')))))
    numErrors=0
    numParsed=0
    for a in list(set(soup.findAll('a'))):
        try:
            url = a['href']
            # if url not already in dictionary and contains not None magicFrag, OR
            # if no magicFrag was called
            if( (url not in urlBodies) and (magicFrag is not None and magicFrag in url) or (magicFrag is None)):
                # then pass the article url,token to the scraperFunction
                # NYTText or getWashPostText for this example
                body = scraperFunction(url,token)
                # handles error/ None return from scraperFunction for parsing error
                # if body != None *and* len(body)!=0
                if body and len(body)>0:
                    urlBodies[url] = body
                    #print('Good Parse for {}...{}'.format(url[:10],url[-10:]))
                    numParsed+=1
        except:
            numErrors+=1
            print('\nBad Parse for \n{}'.format(url))
    print('\nExecution Complete \nscrapeSourceReport: Errors={}, Parsed={}\n'.format(numErrors,numParsed))
    return urlBodies
    
def CorpusCleanup(corpusDict):
    '''input '''
    TechCorpus=[]
    NonTechCorpus=[]
    for k in corpusDict:
        if corpusDict[k]=='Tech':
            TechCorpus.append(k)
        elif corpusDict[k]=='NonTech':
            NonTechCorpus.append(k)

    TechCorpus = ' '.join(TechCorpus)
    NonTechCorpus = ' '.join(NonTechCorpus)

    fs = FrequencySummarizer()
    NonTechWords=fs.extractFeatures((NonTechCorpus,''),1000)
    TechWords=fs.extractFeatures((TechCorpus,''),1000)

    # will combine text+title into a single string as an "article" for this
    #MixedCorpusList = list(''.join(t) for t in MixedCorpus.values() if len(t[0])>0)

    AlmostStopWords=list(set(NonTechWords).intersection(set(TechWords)))
    print('\nNo. of Almost Stop Words = {}'.format(len(AlmostStopWords)))

    #Let's clean up a bunch of the crud in the results above
    #Remove all numbers first using RegEx
    #Preprocess the articles to remove AlmostStopWords, words which were common between 
    #the training data "Tech" and "NonTech" corpora
    #Using RegEx to remove those words only
    #Then remove punctuation and special characters

    PreProcMCL=[]
    PunctStopWords=list(punctuation)+['“','i’m','–','»','“i']
    for doc in MixedCorpusList:
        #regex to remove all numbers
        doc=re.sub(r'[0-9]+','',doc)
        for sw in AlmostStopWords:
            doc=re.sub((r'\b'+sw+r'\b'),'',doc)
        for p in PunctStopWords:
            doc=doc.replace(p,'')
        PreProcMCL.append(doc)
        
    return PreProcMCL
    
def Report_results(modelFit,corpusList,title=None,words=True,sentences=True):
    ''''''
    print('Report_results module {}'.format(modelFit))
    #looking at results
    keywords = {} #dictionary to store the keywords
    keywordFreq = {} #dictionary to store documents by key=cluster
    for i,cluster in enumerate(modelFit.labels_):
        #print('\ncluster No. = {}'.format(cluster))
        oneDocument = corpusList[i]
        fs = FrequencySummarizer()
        summary = fs.extractFeatures((oneDocument,''),
                                     100)
        for w in summary[:4]:
            pass
            #print(w)
        if cluster not in keywords:
            keywords[cluster] = set(summary)
            keywordFreq[cluster] = MixedCorpusList[i]#'''go back to preprocessed for sentence extraction'''
        else:
            keywords[cluster] = keywords[cluster].intersection(set(summary))
            keywordFreq[cluster] = keywordFreq[cluster]+' '+MixedCorpusList[i]

    # attempt to find words that are common to every document in each cluster (futile for too many doc's)
    print('\nCommon words to every document in a cluster:')
    for k in keywords:
        if len(keywords[k])==0:
            pass #print('    ....no common words....')
        else:
            print('\nCluster = {}'.format(k))
            for word in keywords[k]:
                print(' keyword = {}'.format(word))

    # getting good separation with at least 3 clusters, let's look at most frequent words in each cluster
    # since not seeing common words among such a large number of documents
    if words:
        n=3
        print('\n{} Most Frequent words to each cluster:'.format(n))
        for c in keywordFreq:
            print('\n Cluster = {}'.format(c))
            print(' Keywords:')
            kwds = fs.extractFeatures((keywordFreq[c],''),n)
            for w in kwds:
                print('  '+w)        
            #regex to break up sentences, some spaces removed
            ''''''
            if sentences:
                corpus = keywordFreq[c].replace('.','. ')
                summarize_sents=fs.summarize((corpus,''),1)
                print('\n Key Sentences from Cluster {}:'.format(c))
                for s in summarize_sents:
                    print('\n  '+s[:100]+' ... '+s[-100:]+' <end>')
            
    df = pd.DataFrame({'Source':LabelList, 'modelFit_label':modelFit.labels_})
    fp=sns.factorplot(hue='Source',x='modelFit_label',data=df,kind='count')
    if title:
        a=fp.fig.axes[0]
        a.set_title(title)
    fp.savefig(title.replace(' ','').replace('=','')+'_out.png')
    plt.close('all')
        
class PlotSummaryResults:
    ''''''
    
    def __init__(self):
        self.Inertia={}
        self.Silhouette={}
        self.Results_={}
        
    def _metrics(self,i,modelFit,X):
        try:
            self.Inertia[i]=modelFit.inertia_
        except:
            self.Inertia[i]=np.nan
        self.Silhouette[i]=metrics.silhouette_score(X, modelFit.labels_, metric='euclidean')
        
    def _dataframe(self,modelTest):
        df=pd.DataFrame({'KM Clusters':list(self.Inertia.keys()),'Inertia':list(self.Inertia.values()),'Silhouette':list(self.Silhouette.values())})
        self.Results_[modelTest]=df
        
    def PlotSummary(self,MaxClst):
        df=pd.concat(self.Results_)
        df=df.reset_index(level=0)
        df=df.rename(columns={'level_0':'Model'})
        
        f=plt.figure()
        df=df.sort_values(by='KM Clusters')
        dfg=df.groupby(['Model'])
        for j,m in enumerate(['Inertia','Silhouette']):
            a=f.add_subplot(2,2,int(1+2*j))
            a.set_title(m+' vs '+'KM Clusters')
            for n,g in dfg:
                plt.plot(g['KM Clusters'],g[m],label=n,marker='o')
            a.legend(title='Model',loc=(1.05,0.6),shadow=True, frameon=True)
        f.subplots_adjust(hspace=0.3)
        f.savefig('classFigModelComp{}MaxClusters.png'.format(MaxClst))
        plt.close('all')

if __name__ == '__main__':
    
    #get doxydonkey posts
    doxydonkey=False
    
    #run KMeans clustering on doxydonkey posts
    KM_0=False
    
    #get Washpo and NYT articles
    NewsArticles=False
    
    '''Cluster Detection Algorithms
    Number of Clusters is an Output
    '''
    #DBSCAN Density Based Scan
    DBSCAN_Bool=False #does not appear to find any clusters
    
    MeanShift
    MnShft=False #does not appear to find any clusters

    #Affinity Propagation
    AffProp=True  
    
    '''Clustering Algo's where the cluster number is an Input'''
    #run KM clustering on Washpo and NYT Tech and Non-Tech Corpus to find clusters
    KM_1=False
    
    #Generate almoststopwords from common words between Tech and Non-Tech subset corpora
    #pre-process all documents to remove numbers etc
    KM_2=True
    
    #AgglomerativeClustering
    AggClst=True
    
    #MaxClusters for looping test
    MaxClusters=30
    
    #Increment Size
    ClusterStep=1
    
    # Setup Color palette for the number of Clustering Algo's to overlay
    NumModels=3
    sns.set_palette('hls',NumModels,1)
    
    #Instantiate Summary Results Class
    ps=PlotSummaryResults()
    
    #Pre-processed corpus
    PreProcMCL = CorpusCleanup(MixedCorpusDict)

    if doxydonkey:
        blogUrl = 'http://doxydonkey.blogspot.in'
        links = []
        getAllDoxyDonkeyPosts(blogUrl,links)
        doxyDonkeyPosts = {}
        for link in links:
            doxyDonkeyPosts[link] = getDoxyDonkeyText(link,'post-body')

        # store the article in a list of lists for sklean Tfidfvectorizer
        documentCorpus = []
        for onePost in doxyDonkeyPosts.values():
            documentCorpus.append(onePost[0])
        # number of articles in Corpus

        print('\nNo. of documents in Corpus = {}'.format(len(documentCorpus)))
        
    if NewsArticles:
        #Training Data
        urlWashingtonPostNonTech = 'https://www.washingtonpost.com/sports/'
        urlWashingtonPostTech = 'https://www.washingtonpost.com/business/technology/'
        urlNYTNonTech = 'http://www.nytimes.com/pages/sports/index.html'
        urlNYTTech = 'http://www.nytimes.com/pages/technology/index.html'
    
        washPoTechArticles = scrapeSource(urlWashingtonPostTech,'2016',getWashPostText,'article')
        washPoNonTechArticles = scrapeSource(urlWashingtonPostNonTech,'2016',getWashPostText,'article')
        NYTTechArticles = scrapeSource(urlNYTTech,'2016',getNYTText,None)
        NYTNonTechArticles = scrapeSource(urlNYTNonTech,'2016',getNYTText,None)
    
    
        # Success finally, having issues with WashPo rejecting my requests ...
        # time to collect these article summaries in an easy to classify form
        # training data, labeling with Tech and NonTech
        # generating a summary for each article
        articleSumm = {}
        for techURLdict in [washPoTechArticles,NYTTechArticles]:
            for articleURL in techURLdict:
                if techURLdict[articleURL][0] is not None and len(techURLdict[articleURL][0])>0: 
                    fs = FrequencySummarizer()
                    summary = fs.extractFeatures(techURLdict[articleURL],25)
                    articleSumm[articleURL]= {'feature-vector': summary, 'label': 'Tech'}
                
        for NONtechURLdict in [washPoNonTechArticles,NYTNonTechArticles]:
            for articleURL in NONtechURLdict:
                if NONtechURLdict[articleURL][0] is not None and len(NONtechURLdict[articleURL][0])>0: 
                    fs = FrequencySummarizer()
                    summary = fs.extractFeatures(NONtechURLdict[articleURL],25)
                    articleSumm[articleURL]= {'feature-vector': summary, 'label': 'NonTech'}
    
    
        print('\nLength of Article Corpus = {}'.format(len(articleSumm)))
    
    
        # Combine the Tech and Non-Tech dictionaries for ease of use
        TechArts={}
        NonTechArts={}
    
        TechArts.update(washPoTechArticles)
        TechArts.update(NYTTechArticles)
    
        NonTechArts.update(washPoNonTechArticles)
        NonTechArts.update(NYTNonTechArticles)
    
        # create a single List of Articles
        # combine 1 more time to create a single corpus
        # reminder this is a dictionary with url's as keys values are a tuple of (text,title)
        MixedCorpusList = []
        MixedCorpusDict = {}
        TechCount=0
        NonCount=0
        for k in NonTechArts:
            if len(NonTechArts[k][0])>0:
                MixedCorpusList.append(NonTechArts[k][0])
                NonCount+=1
                MixedCorpusDict[NonTechArts[k][0]]='NonTech'
    
        for k in TechArts:
            if len(TechArts[k][0])>0:
                MixedCorpusList.append(TechArts[k][0])
                TechCount+=1
                MixedCorpusDict[TechArts[k][0]]='Tech'
            
        print('No. Tech Articles Counted with >0 Characters = {}\nNo. NonTech Articles Counted with >0 Characters = {}'.format(TechCount,NonCount))
        print('\nTotal No. Articles with >0 Characters = {}'.format(len(MixedCorpusList)))
        print('Total No. of Unique Articles with >0 Characters = {}'.format(len(MixedCorpusDict)))
        print('\n !That means there are {} non-unique articles that appear in both Tech and NonTech Dictionaries!'.format(len(MixedCorpusList)-len(MixedCorpusDict)))
        print('Need to drop these from corpus!')
    
        valuCount=defaultdict(int)
        for k in MixedCorpusDict:
            valuCount[MixedCorpusDict[k]]+=1
    
        print('\nNo. unique Tech Articles = {}'.format(valuCount['Tech']))
        print('No. unique NonTech Articles = {}'.format(valuCount['NonTech']))
    
        #Let's use the unique values in MixedCorpusDict to re-initialize the MixedCorpusList and keep a LabelList to accompany:
        MixedCorpusList = []
        LabelList = []
        for k in MixedCorpusDict:
            MixedCorpusList.append(' '+k)
            LabelList.append(MixedCorpusDict[k])
    
        print('\nRevised Total No. Articles with >0 Characters = {}'.format(len(MixedCorpusList)))
        print('dropped the duplicate articles that appeared in both NonTech and Tech lists')
    
    
        # double check for short articles (0 length were initially found, but fixed that above in building MixedCorpusList)
        shorties=[]
        print('Starting length of MixedCorpusList = {}\n'.format(len(MixedCorpusList)))
        for art in MixedCorpusList:
            if len(art)<200: 
                print('Removed Short Article: \n '+art)
                shorties.append(art)
        for shart in shorties:
            MixedCorpusList.remove(shart)
        print('\nFinal length of MixedCorpusList = {}'.format(len(MixedCorpusList)))    
        #print(MixedCorpusList[2][:100]+'\n  ...***...  \n'+MixedCorpusList[2][-100:-1])
       

    if KM_0 and len(documentCorpus)>1:
        
        KM_clusters=5
        
        # Define the vectorizer model
        vectorizer = TfidfVectorizer(max_df=0.9,min_df=0.1,stop_words='english')
    
        # Vectorize the training data corpus
        X = vectorizer.fit_transform(documentCorpus)
    
        # Instantiate the KMeans model
        km = KMeans(n_clusters=KM_clusters, init = 'k-means++', max_iter=100, n_init=5, verbose=True)
        
        # Fit the KMeans model to the training data
        print('\nFit model to training data...\n')
        km.fit(X)
        
        #looking at results
        #find the keywords in each cluster of documents
        keywords = {} #dictionary to store the keywords
        for i,cluster in enumerate(km.labels_):
            oneDocument = documentCorpus[i]
            fs = FrequencySummarizer()
            summary = fs.extractFeatures((oneDocument,''),
                                     100,
                                     customStopWords=['according','also','billion','like','new','one','year','first','last','million'])
            
            if cluster not in keywords:
                keywords[cluster] = set(summary)
            else:
                keywords[cluster] = keywords[cluster].intersection(set(summary))
    
        # look at the results
        for k in keywords:
            print('\nCluster = {}'.format(k))
            for word in keywords[k]:
                print('keyword = {}'.format(word))
    
        #get_ipython().magic('matplotlib inline')
    
        plt.hist(km.labels_,bins=5)
        #plt.close()

    
    if KM_1 and len(MixedCorpusList)>1:
        i=1        
        while i <MaxClusters:
            i+=ClusterStep
            print('\nKM_1 with {} Clusters'.format(i))
            KM_clusters=i
            # Define the Tfidf vectorizer model
            vectorizer = TfidfVectorizer(stop_words='english',strip_accents='unicode',max_df=0.9, min_df=0.2)
            # Vectorize the training data corpus
            X = vectorizer.fit_transform(MixedCorpusList)
            # Fit the model to the vectorized training data
            km = KMeans(n_clusters=KM_clusters, init = 'k-means++', max_iter=200, n_init=20, verbose=False)
            print('  Fit model to training data...\n')
            km.fit(X)
            
            ps._metrics(i,km,X)
            
            Report_results(km,MixedCorpusList,'KM_1 with {} Clusters; Inertia = {:.1f}'.format(km.n_clusters,km.inertia_),sentences=False,words=False)
        
        ps._dataframe('KM_1')
    

    if KM_2 and len(MixedCorpusList)>1:
        
        i=1 
        while i <MaxClusters:
            i+=ClusterStep
            print('\nKM_2 with {} Clusters'.format(i))
            KM_clusters=i                         
            
            # Define the Tfidf vectorizer model
            vectorizer = TfidfVectorizer(stop_words='english',strip_accents='unicode',max_df=0.9,min_df=0.2)
            # Vectorize the training data corpus
            X = vectorizer.fit_transform(PreProcMCL)
            # Fit the model to the vectorized training data
            km = KMeans(n_clusters=KM_clusters, init = 'k-means++', max_iter=200, n_init=100, verbose=False)
            
            print('  Fit model to training data...\n')
            km.fit(X)

            ps._metrics(i,km,X)
            
            Report_results(km,PreProcMCL,'KM_2 with {} Clusters; Inertia = {:.1f}'.format(km.n_clusters,km.inertia_),sentences=False,words=False)
                     
        ps._dataframe('KM_2')
            
    if AggClst and len(MixedCorpusList)>1:
        i=1 
        while i <MaxClusters:
            i+=ClusterStep
            print('\n AgglomerativeClustering with {} Clusters'.format(i))
            KM_clusters=i                         
            
            # Define the Tfidf vectorizer model
            vectorizer = TfidfVectorizer(stop_words='english',strip_accents='unicode',max_df=0.9,min_df=0.2)
            # Vectorize the training data corpus
            X = vectorizer.fit_transform(PreProcMCL)
            #             
            X = X.toarray()
            # Fit the model to the vectorized training data
            agc = AgglomerativeClustering(n_clusters=KM_clusters)
            
            print('  Fit model to training data...\n')
            agc.fit(X)

            ps._metrics(i,agc,X)
            
            Report_results(agc,PreProcMCL,'Agglomerative Clustering with {} Clusters'.format(agc.n_clusters),sentences=False,words=False)

        ps._dataframe('AggCluster')
                
    if KM_1 or KM_2 or AggClst:
        ps.PlotSummary(MaxClusters)
        
    if DBSCAN_Bool and len(MixedCorpusList)>1:
        
        # Define the Tfidf vectorizer model
        vectorizer = TfidfVectorizer(stop_words='english',strip_accents='unicode',max_df=0.9,min_df=0.2)
        # Vectorize the training data corpus
        X = vectorizer.fit_transform(PreProcMCL)
        #
        #X = StandardScaler(with_mean=False).fit_transform(X)
        # Instantiate the model
        dbs = DBSCAN(eps=0.3,min_samples=10)
        
        print('  Fit model to training data...\n')
        # Fit the model to the training data
        dbs.fit(X)
        
        metrics.silhouette_score(X, dbs.labels_, metric='euclidean')
        
    if AffProp and len(MixedCorpusList)>1:
        
        # Define the Tfidf vectorizer model
        vectorizer = TfidfVectorizer(stop_words='english',strip_accents='unicode',max_df=0.9,min_df=0.2)
        # Vectorize the training data corpus
        X = vectorizer.fit_transform(PreProcMCL)
        # Instantiate the model
        af = AffinityPropagation(verbose=True)
        
        print('\n  Fit model to training data...\n')
        # Fit the model to the training data
        af.fit(X)
        
        silh=metrics.silhouette_score(X, af.labels_, metric='euclidean')
        Report_results(af,PreProcMCL,'AffinityProp {} Clusters Silh = {:.1f}'.format(len(set(af.labels_)),silh),sentences=False,words=False)
        
    if MnShft and len(MixedCorpusList)>1:
        
        # Define the Tfidf vectorizer model
        vectorizer = TfidfVectorizer(stop_words='english',strip_accents='unicode',max_df=0.9,min_df=0.2)
        # Vectorize the training data corpus
        X = vectorizer.fit_transform(PreProcMCL)
        # 
        X = X.toarray()
        # Instantiate the model
        ms = MeanShift()
        
        print('\n  Fit model to training data...\n')
        # Fit the model to the training data
        ms.fit(X)
        
        silh=metrics.silhouette_score(X, ms.labels_, metric='euclidean')
        Report_results(ms,PreProcMCL,'MeanShift {} Clusters Silh = {:.1f}'.format(len(set(ms.labels_)),silh),sentences=False,words=False)
    

print('\n... Execution Complete ...')