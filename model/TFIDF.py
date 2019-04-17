#!/usr/bin/env python
# coding: utf-8

# In[18]:


# load packages

import boto3
import botocore
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
nltk.download('punkt')

# Gensim uses Python’s standard logging module to log various stuff at various priority levels; to activate logging (this is optional), run
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


# # Let's use Amazon S3
# client = boto3.client('s3')
# s3 = boto3.resource('s3')
# 
# # Print out bucket names
# for bucket in s3.buckets.all():
#     print(bucket.name)
# bucket_name = 'cse6242oan-xchen668'

# In[5]:


# connect psql server
# psql --host cse6242project.cnsmcycpnqu7.us-east-1.rds.amazonaws.com --p --port 5432 --username=<your_name> --dbname=cse6242project
engine = sqlalchemy.create_engine('postgresql+psycopg2://xchen668:password@cse6242project.cnsmcycpnqu7.us-east-1.rds.amazonaws.com/cse6242project')

# business = pd.read_sql_query("SELECT * FROM {};".format("business"), engine)
businessDf = pd.read_sql_table("business", engine)

# check data schema
businessDf.head()
# drop geom col for postGis
businessDf.drop("geom", axis = 1)


# In[231]:


import re
import string

def get_data(city_name):
    query = """
    select a.*, b.city, b.categories, b.postal_code
    from review a
    inner join 
    business b
    on a.business_id = b.business_id
    where b.is_us = 1
    and b.is_restaurant = 1
    and b.city = '{0}';
    """
    city_res_rev = pd.read_sql_query(query.format(city_name), engine)
    return city_res_rev

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


# In[212]:


phx_res_rev = get_data('Phoenix')


# In[217]:


# phx_res_rev = process_city_reviews(phx_res_rev, 'Mexican')
# phx_res_rev[:5]


# In[226]:


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


# In[227]:


cv, tfidf_transformer = tfidf_vec_fit(phx_res_rev, stem = True, max_df = 0.8)


# In[262]:


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

def tf_idf_trans(data, sentiment, category, postal_code, top_n, cv = cv, tfidf_transformer = tfidf_transformer):
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


# In[229]:


pos_phx_mex_key = tf_idf_trans(phx_res_rev, 'positive', 'Mexican', '85013', 10, cv = cv, tfidf_transformer = tfidf_transformer)
print(pos_phx_mex_key)


# In[230]:


neg_phx_mex_key = tf_idf_trans(phx_res_rev, 'negative', 'Mexican', '85013', 10, cv = cv, tfidf_transformer = tfidf_transformer)
print(neg_phx_mex_key)


# In[232]:


phx_res_rev_amrican = process_city_reviews(phx_res_rev, 'American')
cv, tfidf_transformer = tfidf_vec_fit(phx_res_rev_amrican, stem = True, max_df = 0.8)


# In[241]:


pos_phx_ame_key = tf_idf_trans(phx_res_rev_amrican, 'positive', 'American', '85013', 10, cv = cv, tfidf_transformer = tfidf_transformer)
print(pos_phx_ame_key)


# In[242]:


neg_phx_ame_key = tf_idf_trans(phx_res_rev_amrican, 'negative', 'American', '85013', 10, cv = cv, tfidf_transformer = tfidf_transformer)
print(neg_phx_ame_key)


# In[334]:


def get_key_word(city_data, city, category, zipcode, top_n, stem = True, max_df = 0.8):
    city_data_c = process_city_reviews(city_data, category)
    cv, tfidf_transformer = tfidf_vec_fit(city_data_c, stem = stem, max_df = max_df)
    pos_key = tf_idf_trans(city_data_c, 'positive', category, zipcode, top_n, cv = cv, tfidf_transformer = tfidf_transformer)
    neg_key = tf_idf_trans(city_data_c, 'negative', category, zipcode, top_n, cv = cv, tfidf_transformer = tfidf_transformer)
    return {'city': city, 'category': category, 'zipcode': zipcode, 'pos':pos_key, 'neg':neg_key}


# # Phoenix

# In[270]:


city = 'Phoenix'
get_key_word(phx_res_rev, city, 'American', '85004', 10, stem = True, max_df = 0.8)


# In[256]:


get_key_word(phx_res_rev, city, 'Chinese', '85042', 10, stem = True, max_df = 0.8)


# In[269]:


get_key_word(phx_res_rev, city, 'Nightlife', '85013', 10, stem = True, max_df = 0.8)


# In[281]:


get_key_word(phx_res_rev, city, 'Italian', '85012', 10, stem = True, max_df = 0.8)


# In[278]:


phx_res_rev['postal_code'].value_counts()


# ## Cleveland

# In[273]:


city = 'Cleveland'
cle_res_rev = get_data(city)


# In[274]:


cle_res_rev['postal_code'].value_counts()


# In[276]:


cle_res_rev['categories'].value_counts()


# In[283]:


get_key_word(cle_res_rev, city, 'American', '44113', 10, stem = True, max_df = 0.8)


# In[284]:


get_key_word(cle_res_rev, city, 'Chinese', '44113', 10, stem = True, max_df = 0.8)


# In[312]:


get_key_word(cle_res_rev, city, 'Chinese', '44106', 10, stem = True, max_df = 0.8)


# In[314]:


get_key_word(cle_res_rev, city, 'Breakfast', '44114', 10, stem = True, max_df = 0.8)


# In[335]:


test_key = get_key_word(cle_res_rev, city, 'Breakfast', '44114', 10, stem = True, max_df = 0.8)


# In[338]:


",".join(list(test_key['pos'].keys())) 


# In[326]:


processed_text = {'Cleveland':cle_res_rev, 'Phoenix':phx_res_rev}


# # KNN

# In[307]:


query = """
    select public.nearest_k_restaurant({0}, {1}, {2})
    """
long, lat = -112.0685, 33.4528
test_knn = pd.read_sql_query(query.format(long, lat, 20), engine)['nearest_k_restaurant']


# In[308]:


test_knn


# In[319]:


query = """
    select avg(r.pred_star) as pred_star,  sum(r.perc_rank) AS checkin_rank,
    avg(r.pred_star) + sum(r.perc_rank) as popularity 
    FROM
    (select public.nearest_k_restaurant({0}, {1}, {2}) as business_id) a
    inner join predicted_star_rank r
    on a.business_id = r.business_id;
    """
lat, long = 33.5079722,-112.10129119999999
result = engine.execute(query.format(long, lat, 20)).fetchall()[0]
print(result)


# In[324]:


def get_pop_score(long, lat, k):
    query = """
        select avg(r.pred_star) as pred_star,  sum(r.perc_rank) AS checkin_rank,
        avg(r.pred_star) + sum(r.perc_rank) as popularity 
        FROM
        (select public.nearest_k_restaurant({0}, {1}, {2}) as business_id) a
        inner join predicted_star_rank r
        on a.business_id = r.business_id;
        """
    result = engine.execute(query.format(long, lat, k)).fetchall()[0]
    return result[0], result[1], result[2]


# In[325]:


lat, long, k = 33.5079722,-112.10129119999999, 20
print(get_pop_score(long, lat, k))


# # Output

# In[328]:


entry_json = [{'output_id':101,
'zip':'85004',
'city':'Phoenix',
'latitude': 33.4528292,
'longitude':-112.06850270000001,
'category':'American'
},
{'output_id':102,
'zip':'85042',
'city':'Phoenix',
'latitude': 33.3762931,
'longitude':-112.03571369999997,
'category':'Chinese'
},
{'output_id':103,
'zip':'85013',
'city':'Phoenix',
'latitude': 33.3762931,
'longitude':-112.08638740000004,
'category':'Nightlife'
},
{'output_id':105,
'zip':'85012',
'city':'Phoenix',
'latitude': 33.5114334,
'longitude':-112.06850270000001,
'category':'Italian'
},
{'output_id':106,
'zip':'44113',
'city':'Cleveland',
'latitude': 41.4857101,
'longitude':-81.69663059999999,
'category':'American'
},
{'output_id':107,
'zip':'44114',
'city':'Cleveland',
'latitude': 41.5139193,
'longitude':-81.67472950000001,
'category':'Chinese'
},
{'output_id':108,
'zip':'44106',
'city':'Cleveland',
'latitude': 41.5091257,
'longitude':-81.60898729999997,
'category':'Chinese'
},
{'output_id':109,
'zip':'44106',
'city':'Cleveland',
'latitude': 41.5091257,
'longitude':-81.60898729999997,
'category':'Pizza'
},
{'output_id':110,
'zip':'44114',
'city':'Cleveland',
'latitude': 41.5139193,
'longitude':-81.67472950000001,
'category':'Breakfast'
}
]


# In[339]:


def get_final_output(entry, processed_text = processed_text, k = 20):
    pred_star, checkin_rank, pop_score = get_pop_score(entry['longitude'], entry['latitude'], k)
    key_word_dict = get_key_word(processed_text[entry['city']], entry['city'],
                                 entry['category'], entry['zip'], 10, stem = True, max_df = 0.8)
    #pos_rev = ",".join(list(key_word_dict['pos'].keys()))
    output_dict = entry
    output_dict['pop_score'] = pop_score
    output_dict['pos_rev'] = ",".join(list(key_word_dict['pos'].keys()))
    output_dict['neg_rev'] = ",".join(list(key_word_dict['neg'].keys()))
    return output_dict


# In[340]:


get_final_output(entry_json[0], processed_text = processed_text, k = 20)


# In[343]:


output_list = list(map(get_final_output, entry_json))  


# In[344]:


output_list[0]


# In[347]:


# Create MetaData instance
metadata = MetaData(engine, reflect=True)

# Get Table
out_table = metadata.tables['output']
print(out_table)

conn = engine.connect()

# insert multiple data
conn.execute(out_table.insert(),output_list)


# In[369]:


# Register t1, t2 to metadata
out_table_revised_v2 = Table('output_revised_v2', metadata,
           sqlalchemy.Column('output_id',sqlalchemy.Integer, primary_key=True),
           sqlalchemy.Column('zip',sqlalchemy.String),
           sqlalchemy.Column('city',sqlalchemy.String),
           sqlalchemy.Column('latitude',sqlalchemy.Float),
           sqlalchemy.Column('longitude',sqlalchemy.Float),
           sqlalchemy.Column('category',sqlalchemy.String),
           sqlalchemy.Column('pop_score',sqlalchemy.Float),
           sqlalchemy.Column('pred_star',sqlalchemy.Float),
           sqlalchemy.Column('checkin_rank',sqlalchemy.Float),
           sqlalchemy.Column('post_rev',sqlalchemy.JSON),
           sqlalchemy.Column('neg_rev',sqlalchemy.JSON)
                         )

# Create all tables in meta
metadata.create_all()


# In[364]:


def get_final_output_json(entry, processed_text = processed_text, k = 20):
    pred_star, checkin_rank, pop_score = get_pop_score(entry['longitude'], entry['latitude'], k)
    key_word_dict = get_key_word(processed_text[entry['city']], entry['city'],
                                 entry['category'], entry['zip'], 10, stem = True, max_df = 0.8)
    #pos_rev = ",".join(list(key_word_dict['pos'].keys()))
    output_dict = entry
    output_dict['pop_score'] = pop_score
    output_dict['pred_star'] = pred_star
    output_dict['checkin_rank'] = checkin_rank
    output_dict['pos_rev'] = key_word_dict['pos']
    output_dict['neg_rev'] = key_word_dict['neg']
    return output_dict

output_json_list = list(map(get_final_output_json, entry_json))  
output_json_list[0]


out_table_revised.drop(engine)

# insert multiple data
conn.execute(out_table_revised_v2.insert(),output_json_list)

out_table_revised_v2

entry_json[0]['city']


# ## Explore



query = """
    select a.*, b.city, b.categories
    from review a
    inner join 
    business b
    on a.business_id = b.business_id
    where b.is_us = 1
    and b.is_restaurant = 1
    and b.city = 'Phoenix';
"""
phxResReviews =  pd.read_sql_query(query, engine)

print('\nThe first review:\n')
print(phxResReviews["text"][0], '\n')
print(phxResReviews.shape)
print(phxResReviews.columns)


# In[18]:


data = phxResReviews[pd.notnull(phxResReviews['text'])]
print(data.shape)

#size = 100000 #100,000
# size = 1000000
# data = data.sample(frac=1).reset_index(drop=True)
# subdata, restdata = data.iloc[:size], data.iloc[size:]

#subdata.to_csv('review_sub_{}.csv'.format(size), index=False, quoting=3, sep=',', escapechar='\\', encoding='utf-8')


# In[160]:


len(phxResReviews["text"][0])


# In[161]:


import re
def pre_process(text):
    
    # lowercase
    text=text.lower()
    
    #remove tags
    text=re.sub("&lt;/?.*?&gt;"," &lt;&gt; ",text)
    
    # remove special characters and digits
    text=re.sub("(\\d|\\W)+"," ",text)
    
    return text
 
phxResReviews['text'] = phxResReviews['text'].apply(lambda x:pre_process(x))
 
#show an example
print(phxResReviews['text'][20])


# In[162]:


threshold = np.percentile(phxResReviews["text"].apply(lambda x:len(x)), 10)
print(threshold)


# In[163]:


phxResReviewsT = phxResReviews[phxResReviews['text'].apply(lambda x: len(x)> threshold)]
print(phxResReviewsT.shape)


# In[164]:


from sklearn.feature_extraction.text import CountVectorizer
#get the text column 
docs=phxResReviewsT['text'][phxResReviewsT["categories"].str.contains('Mexican')].tolist()

#create a vocabulary of words, 
#ignore words that appear in 85% of documents, 
#eliminate stop words
cv=CountVectorizer(max_df=0.60,stop_words='english', max_features=10000)
word_count_vector=cv.fit_transform(docs)


# In[166]:


docs[1]


# In[139]:


from sklearn.feature_extraction.text import TfidfTransformer
 
tfidf_transformer=TfidfTransformer(norm = 'l2', smooth_idf=False,use_idf=True)
tfidf_transformer.fit(word_count_vector)


# In[140]:


phxResReviews[:5]


# In[129]:


phxResReviewsCat = phxResReviews.merge(businessDf[['business_id', 'postal_code']], on = "business_id", how = "inner")


# In[130]:


phxResReviewsCat


# In[100]:


mexPhxRev = phxResReviewsCat[(phxResReviewsCat['categories'].str.contains('Mexican'))&
                            (phxResReviewsCat['stars'] > 3)&
                            (phxResReviewsCat['postal_code'] == '85013')]
print(mexPhxRev.shape)


# In[94]:


mexPhxRev


# In[95]:


# get test docs into a list
mexPhxRevT =mexPhxRev['text'].apply(lambda x:pre_process(x))
docs_test=mexPhxRevT.tolist()


# In[96]:


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


# In[98]:


from functools import reduce
# you only needs to do this once, this is a mapping of index to 
feature_names=cv.get_feature_names()
 
# get the document that we want to extract keywords from
doc=reduce(lambda a,b: a + " " + b, docs_test)
 
#generate tf-idf for the given document
tf_idf_vector=tfidf_transformer.transform(cv.transform([doc]))
 
#sort the tf-idf vectors by descending order of scores
sorted_items=sort_coo(tf_idf_vector.tocoo())
 
#extract only the top n; n here is 10
keywords=extract_topn_from_vector(feature_names,sorted_items,10)
 
# now print the results
print("\n=====Doc=====")
#print(doc)
print("\n===Keywords===")
for k in keywords:
    print(k,keywords[k])


# In[106]:





# In[143]:


neg_phx_mex_key = tf_idf_trans(phxResReviewsCat, 'negative', 'Mexican', '85012', 10)
print(neg_phx_mex_key)


# In[144]:


pos_phx_mex_key = tf_idf_trans(phxResReviewsCat, 'positive', 'Mexican', '85012', 10)
print(pos_phx_mex_key)

