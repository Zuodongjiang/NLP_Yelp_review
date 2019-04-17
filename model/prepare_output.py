import psycopg2
import sqlalchemy
from sqlalchemy import MetaData
#from sqlalchemy import Table

import json
import pandas as pd
import numpy as np
import time
import copy
import pickle
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from functools import reduce
import string

from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize

import keywords_utils as utils
#nltk.download('punkt')

# connect psql server
# psql --host cse6242project.cnsmcycpnqu7.us-east-1.rds.amazonaws.com --p --port 5432 --username=<your_name> --dbname=cse6242project
engine = sqlalchemy.create_engine('postgresql+psycopg2://xchen668:password@cse6242project.cnsmcycpnqu7.us-east-1.rds.amazonaws.com/cse6242project')

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

cities = ['Cleveland', 'Phoenix']
processed_text = dict()

for c in cities:
	res_rev_df = get_data(c)
	processed_text[c] = res_rev_df

# eg. pos_phx_mex_key = tf_idf_trans(phx_res_rev, 'positive', 'Mexican', '85013', 10, cv = cv, tfidf_transformer = tfidf_transformer)
def get_key_word(city_data, city, category, zipcode, top_n, stem = True, max_df = 0.8):
    city_data_c = utils.process_city_reviews(city_data, category)
    cv, tfidf_transformer = utils.tfidf_vec_fit(city_data_c, stem = stem, max_df = max_df)
    pos_key = utils.tf_idf_trans(city_data_c, 'positive', category, zipcode, top_n, cv = cv, tfidf_transformer = tfidf_transformer)
    neg_key = utils.tf_idf_trans(city_data_c, 'negative', category, zipcode, top_n, cv = cv, tfidf_transformer = tfidf_transformer)
    return {'city': city, 'category': category, 'zipcode': zipcode, 'pos':pos_key, 'neg':neg_key}

# eg. city = 'Phoenix'
# get_key_word(phx_res_rev, city, 'American', '85004', 10, stem = True, max_df = 0.8)

### get popularity score
### Notice: sum(r.perc_rank)*20/k is to fix the calculation of perc_rank which is base on k = 20.
def get_pop_score(long, lat, k, category):
    query = """
        select avg(r.pred_star) as pred_star,  sum(r.perc_rank)*20/{2} AS checkin_rank,
        avg(r.pred_star) + sum(r.perc_rank)*20/{2} as popularity 
        FROM
        (select public.nearest_k_restaurant_v2({0}, {1}, {2}, '{3}') as business_id) a
        inner join predicted_star_rank r
        on a.business_id = r.business_id;
        """
    result = engine.execute(query.format(long, lat, k, category)).fetchall()[0]
    return result[0], result[1], result[2]

def get_final_output(entry, processed_text = processed_text, k = 10):
    t0 = time.time()
    pred_star, checkin_rank, pop_score = get_pop_score(entry['longitude'], entry['latitude'], k, entry['category'])
    key_word_dict = get_key_word(processed_text[entry['city']], entry['city'],
                                 entry['category'], entry['zip'], 10, stem = True, max_df = 0.8)
    #pos_rev = ",".join(list(key_word_dict['pos'].keys()))
    output_dict = copy.deepcopy(entry)
    output_dict['pop_score'] = pop_score
    output_dict['pos_rev'] = ",".join(list(key_word_dict['pos'].keys()))
    output_dict['neg_rev'] = ",".join(list(key_word_dict['neg'].keys()))
    
    output_js_dict = copy.deepcopy(entry)
    output_js_dict['pop_score'] = pop_score
    output_js_dict['pred_star'] = pred_star
    output_js_dict['checkin_rank'] = checkin_rank
    output_js_dict['pos_rev'] = key_word_dict['pos']
    output_js_dict['neg_rev'] = key_word_dict['neg']
    t1 = time.time()
    print("{0}, {1}, {2} is complete: {3}s".format(entry['city'], entry['zip'], entry['category'], t1-t0))
    return (output_dict,output_js_dict)

with open('./data/entry.json', 'r') as f:
    entry_json = json.load(f)

#output_list = list(map(get_final_output, entry_json[0]))  
output_list = list()
output_js_list = list()
for entry in entry_json:
    try:
        output1,output2 = get_final_output(entry, processed_text = processed_text, k = 5)
    except Exception as error:
        print(error)
        continue
    output_list.append(output1)
    output_js_list.append(output2)

print(output_js_list[100])
print(output_js_list[0])

# Create MetaData instance
metadata = MetaData(engine, reflect=True)

# Get Table
out_table = metadata.tables['output']
print(out_table)

conn = engine.connect()

# insert multiple data
conn.execute(out_table.insert(),output_list)
"""
def get_final_output_json(entry, processed_text = processed_text, k = 10):
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
    print("{0}, {1} has finished: time spend {2}s".format())
    return output_dict

output_json_list = list(map(get_final_output_json, entry_json))  
output_json_list[0]
"""
# Get Table
out_table_revised_v2 = metadata.tables['output_revised_v2']
print(out_table_revised_v2)

# insert multiple data
conn.execute(out_table_revised_v2.insert(),output_js_list)

"""
out_table_revised_v2.drop(engine)

# Register output_revised_v2 to metadata
out_table_revised_v2 = sqlalchemy.Table('output_revised_v2', metadata,
           sqlalchemy.Column('output_id',sqlalchemy.Integer, primary_key=True),
           sqlalchemy.Column('zip',sqlalchemy.String),
           sqlalchemy.Column('city',sqlalchemy.String),
           sqlalchemy.Column('latitude',sqlalchemy.Float),
           sqlalchemy.Column('longitude',sqlalchemy.Float),
           sqlalchemy.Column('category',sqlalchemy.String),
           sqlalchemy.Column('pop_score',sqlalchemy.Float),
           sqlalchemy.Column('pred_star',sqlalchemy.Float),
           sqlalchemy.Column('checkin_rank',sqlalchemy.Float),
           sqlalchemy.Column('pos_rev',sqlalchemy.JSON),
           sqlalchemy.Column('neg_rev',sqlalchemy.JSON)
                         )

# Create all tables in meta
metadata.create_all()

with open('./data/output_list_v2.pk', 'wb') as f:
    pickle.dump(output_list, f)
    
with open('./data/output_js_list_v2.pk', 'wb') as f:
    pickle.dump(output_js_list, f)
"""