import requests,sys
import os
import json
import datetime
import time
import pytz
from confluent_kafka import Producer

from dotenv import load_dotenv
load_dotenv() #load .env file

#Kafka configuration
topic = os.environ['KAFKA_TOPIC']
conf = {
        'bootstrap.servers': os.environ['KAFKA_BROKERS'],
       # 'session.timeout.ms': 6000,
       # 'default.topic.config': {'auto.offset.reset': 'smallest'},
        'security.protocol': 'SASL_SSL',
	    'sasl.mechanisms': 'SCRAM-SHA-256',
        'sasl.username': os.environ['KAFKA_USERNAME'],
        'sasl.password': os.environ['KAFKA_PASSWORD']
    }

p = Producer(**conf)

#Twitter 
bearer_token = os.getenv('TwitterAPI_BEARER_TOKEN')
keyword = "new york city lang:en" #Tweets search Query


def create_headers(bearer_token):
    headers = {"Authorization": "Bearer {}".format(bearer_token)}
    return headers

def create_url(keyword, start_date, end_date, max_results = 10):
    search_url = "https://api.twitter.com/2/tweets/search/recent" 
    query_params = {'query': keyword,
                    'start_time': start_date,
                    'end_time': end_date,
                    'max_results': max_results,
                    'tweet.fields': 'created_at,source',
                   # 'expansions': 'author_id,in_reply_to_user_id,geo.place_id',
                   # 'tweet.fields': 'id,text,author_id,in_reply_to_user_id,geo,conversation_id,created_at,lang,public_metrics,referenced_tweets,reply_settings,source',
                   # 'user.fields': 'id,name,username,created_at,description,public_metrics,verified',
                   # 'place.fields': 'full_name,id,country,country_code,geo,name,place_type',
                    'next_token': {}}
    return (search_url, query_params)

def connect_to_endpoint(url, headers, params, next_token = None):
    params['next_token'] = next_token 
    response = requests.request("GET", url, headers = headers, params = params)
    print("Endpoint Response Code: " + str(response.status_code))
    if response.status_code != 200:
        raise Exception(response.status_code, response.text)
    return response.json()

def AddtoKafka(record):
    p.produce(topic, json.dumps(record))
    p.flush()
    #print(record)

def formatResponse(json_response):
    for item in json_response["data"]:
       # print(item)
        record={}
        record["text"]=item["text"]
        record["source"]=item["source"]
        record["created_at"]=item["created_at"]
        AddtoKafka(record)
    

newYorkTz   = pytz.timezone("America/New_York") 
endtime     = datetime.datetime.now(newYorkTz) - datetime.timedelta(minutes = 1) 
starttime   = endtime - datetime.timedelta(minutes = 15) 
start_time = starttime.isoformat()
end_time = endtime.isoformat()
print("Tweets between :",start_time," to ",end_time)

headers  = create_headers(bearer_token)
max_results = 15
count = 0 
max_count = 35 
flag = True
next_token = None
reqCount=0

while flag:
    if count >= max_count:
        break
    print("-------------------")
    print("Token: ", next_token)
    url = create_url(keyword, start_time,end_time, max_results)
    json_response = connect_to_endpoint(url[0], headers, url[1], next_token)
    result_count = json_response['meta']['result_count']

    if 'next_token' in json_response['meta']:
        next_token = json_response['meta']['next_token']
        if result_count is not None and result_count > 0 and next_token is not None:
            count += result_count
            formatResponse(json_response)
            reqCount +=1
            print("Request Count :",reqCount,"Number of tweets :",count)
            print("-------------------")
            time.sleep(1)                
    else:
        if result_count is not None and result_count > 0:
            count += result_count
            reqCount +=1
            formatResponse(json_response)
            print("Request Count :",reqCount,"Number of tweets :",count)
            print("-------------------")
        flag = False
        next_token = None
print("Final number of results: ", count)