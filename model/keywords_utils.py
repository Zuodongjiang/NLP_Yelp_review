import psycopg2
import sqlalchemy
from sqlalchemy import MetaData
from sqlalchemy import Table

import pandas as pd
import numpy as np
import time
import pickle
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from functools import reduce
import string

from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize

# clean text data
def pre_process(text):
    
    # lowercase
    text=text.lower()
    
    #remove tags
    #text=re.sub("&lt;/?.*?&gt;"," &lt;&gt; ",text)
    
    # remove punctuations and digits
    #text=re.sub("(\\d|\\W)+"," ",text)
    text = re.sub("(\\d)+"," ",text)
    text = ' '.join(word.strip(string.punctuation) for word in text.split())
    return text

def process_city_reviews(city_res_rev, category):
    city_res_rev = city_res_rev[city_res_rev['categories'].str.contains(category)]
    city_res_rev = city_res_rev[pd.notnull(city_res_rev['text'])]
    city_res_rev['text'] = city_res_rev['text'].apply(lambda x:pre_process(x))
    return city_res_rev

def tfidf_vec_fit(data, stem = False, percentile = 10, max_features = 10000, max_df = 0.60):
    threshold = np.percentile(data["text"].apply(lambda x:len(x)), percentile)
    data_t = data['text'][data['text'].apply(lambda x: len(x)> threshold)]
    docs = data_t.tolist()

    # create a vocabulary of words, 
    # ignore words that appear in 55% of documents, 
    # eliminate stop words
    
    cv=CountVectorizer(max_df=max_df,stop_words='english', max_features=max_features)
    
    if stem:
        stemmer = SnowballStemmer('english')
        def token_stem(doc):
            tokens = word_tokenize(doc)
            stemmed_tokens = list(map(stemmer.stem, tokens))
            stemmed_doc = " ".join(stemmed_tokens)
            return stemmed_doc
        docs = list(map(token_stem, docs))
        
    word_count_vector=cv.fit_transform(docs)
    tfidf_transformer=TfidfTransformer(norm = 'l2', smooth_idf=False,use_idf=True)
    tfidf_transformer.fit(word_count_vector)
    return (cv, tfidf_transformer)

def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)
 
def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """get the feature names and tf-idf score of top n items"""
    
    #use only topn items from vector
    sorted_items = sorted_items[:topn]
 
    score_vals = []
    feature_vals = []
    
    # word index and corresponding tf-idf score
    for idx, score in sorted_items:
        
        #keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])
 
    #create a tuples of feature,score
    #results = zip(feature_vals,score_vals)
    results= {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]]=score_vals[idx]
    
    return results

def tf_idf_trans(data, sentiment, category, postal_code, top_n, cv, tfidf_transformer):
    if sentiment == 'positive':
        sub_data = data[(data['categories'].str.contains(category))&
                                (data['stars'] > 3)&
                                (data['postal_code'] == postal_code)]
    elif sentiment == 'negative':
        sub_data = data[(data['categories'].str.contains(category))&
                                (data['stars'] < 3)&
                                (data['postal_code'] == postal_code)]  
    else:
        raise Exception("Wrong Entry for sentiment")

    # get sub docs into a list
    sub_data_t =sub_data['text'].apply(lambda x:pre_process(x))
    docs_test=sub_data_t.tolist()   

    # you only needs to do this once, this is a mapping of index to 
    feature_names=cv.get_feature_names()
     
    # get the document that we want to extract keywords from
    doc=reduce(lambda a,b: a + " " + b, docs_test)
     
    #generate tf-idf for the given document
    tf_idf_vector=tfidf_transformer.transform(cv.transform([doc]))
     
    #sort the tf-idf vectors by descending order of scores
    sorted_items=sort_coo(tf_idf_vector.tocoo())
     
    #extract only the top n; n here is 10
    keywords=extract_topn_from_vector(feature_names,sorted_items,top_n)
    return keywords