import sys
import os
import json

from dotenv import load_dotenv
load_dotenv()

from confluent_kafka import Consumer, KafkaException, KafkaError
from datetime import datetime

from nltk.corpus import stopwords
import nltk
from typing import *

import re
import pyspark
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType, FloatType, IntegerType
from pyspark.ml.functions import vector_to_array
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml.classification import LogisticRegressionModel

from pymongo import MongoClient
from datetime import datetime
import uuid
import random
from pprint import pprint



topics = [os.environ['KAFKA_TOPIC']]
kafka_conf= {
        'bootstrap.servers': os.environ['KAFKA_BROKERS'],
        'group.id': "%s-consumer" % os.environ['KAFKA_USERNAME'],
        'session.timeout.ms': 6000,
        'default.topic.config': {'auto.offset.reset': 'smallest'},
        'security.protocol': 'SASL_SSL',
	    'sasl.mechanisms': 'SCRAM-SHA-256',
        'sasl.username': os.environ['KAFKA_USERNAME'],
        'sasl.password': os.environ['KAFKA_PASSWORD']
    }

# init spark
conf = pyspark.SparkConf()
conf.set('spark.sql.repl.eagerEval.enabled', True)
conf.set('spark.driver.memory','6g')
sc = pyspark.SparkContext(conf=conf)
sc.setLogLevel("OFF")
spark = pyspark.SQLContext.getOrCreate(sc)

# nltk stopword path
NLTK_STOPWORD_PATH = "nltk_data"
MODEL_PATH = "lr_bestModel"


def get_hate_demo_samples():
    hate_samples = list(json.load(open("HateSpeech.json")).items())
    hate_samples = random.sample(hate_samples, 100) # get 100 samples
    ids, tweets, time_stamps = [], [], []
    for sample in hate_samples:
        ids.append(str(uuid.uuid1()))
        tweets.append(sample[1])
        time_stamps.append(datetime.now().strftime("%Y-%m-%dT%H:%M:%S.000Z"))

    return ids, tweets, time_stamps


def get_data(data_samples):
    ids = []
    tweets = []
    time_stamps = []
    for data in data_samples:
        ids.append(data["id"])
        tweets.append(data["text"])
        time_stamps.append(data["created_at"])
    
    return ids, tweets, time_stamps


def preprocess_data(ids: List[str], tweets: List[str], time_stamps: List[str]):
    # create pyspark df
    column_names = ["id", "tweet", "time_stamp"]
    data = list(zip(ids, tweets, time_stamps))
    df = spark.createDataFrame(data=data, schema=column_names)

    # remove stopwords and special char
    nltk.data.path.append(NLTK_STOPWORD_PATH)
    stopwords_list = stopwords.words('english')
    stopwords_list.extend(["amp"])

    @udf(returnType=StringType())
    def get_cleaned_tweet(tweet):
        tweet = re.sub(r'http\S+', '', tweet) # remove urls
        tweet = re.sub(r'[a-z0-9\.\-+_]+@[a-z0-9\.\-+_]+\.[a-z]+', '', tweet) # remove emails
        tweet = re.sub('[^a-zA-Z0-9]', " ", tweet) # replace special chars with white space
        tweet = re.sub(r'\s+', ' ', tweet) # replace whitespace(s) with a single space
        tweet = re.sub(r'^\s+|\s+?$', '', tweet) # remove leading and trailing whitespace
        tweet = re.sub(r'\d+(\.\d+)?', '[num]', tweet) # replace all numbers with string `num`
        tweet = tweet.lower() # lowercase
        tweet = ' '.join([word for word in tweet.split() if not word in stopwords_list])
        
        return tweet

    @udf(returnType=StringType())
    def get_cleaned_tweet_wo_hashtag(tweet):
        tweet = re.sub("#[A-Za-z0-9_]+","", tweet)
        tweet = re.sub(r'http\S+', '', tweet) # remove urls
        tweet = re.sub(r'[a-z0-9\.\-+_]+@[a-z0-9\.\-+_]+\.[a-z]+', '', tweet) # remove emails
        tweet = re.sub('[^a-zA-Z0-9]', " ", tweet) # replace special chars with white space
        tweet = re.sub(r'\s+', ' ', tweet) # replace whitespace(s) with a single space
        tweet = re.sub(r'^\s+|\s+?$', '', tweet) # remove leading and trailing whitespace
        tweet = re.sub(r'\d+(\.\d+)?', '[num]', tweet) # replace all numbers with string `num`
        tweet = tweet.lower() # lowercase
        tweet = ' '.join([word for word in tweet.split() if not word in stopwords_list])

        return tweet

    df = df.withColumn("tweet_wo_hashtag", get_cleaned_tweet_wo_hashtag("tweet"))
    df = df.withColumn("tweet", get_cleaned_tweet("tweet"))

    return df


def transform_data(df):
    # tokenization
    tokenizer = Tokenizer(inputCol="tweet", outputCol="words")
    wordsData = tokenizer.transform(df)

    # term frequency
    hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures") # default numFeatures: int=262144 to avoid collision
    featurizedData = hashingTF.transform(wordsData)

    # inverse document frequency
    idf = IDF(inputCol="rawFeatures", outputCol="features")
    idfModel = idf.fit(featurizedData)
    rescaledData = idfModel.transform(featurizedData)

    return rescaledData


def process_prediction(predictions):
    # formate the prediction
    predictions = predictions.select(["id", "tweet", "tweet_wo_hashtag", "time_stamp", "prediction", "probability"]) # filter out rawPrediction
    predictions = predictions.withColumnRenamed("probability", "confidence_score") # rename

    # change prediction to str and confidence_score to list[float], also map prediction to class in str
    mapping = {"0":"not_hate", "1":"hate"}
    predictions = predictions.withColumn(
        "prediction", col("prediction").cast(IntegerType()).cast(StringType())
    ).withColumn(
        "confidence_score", vector_to_array("confidence_score")
    ).replace(to_replace=mapping, subset=["prediction"])

    # filter out confidence_score to only keep the one correspond to predicted class
    @udf(returnType=FloatType())
    def get_cls_confidence(prediction, confidence_score):
        if prediction == "not_hate":
            return confidence_score[0]
        else:
            return confidence_score[1]
        
    predictions = predictions.withColumn("confidence_score", get_cls_confidence("prediction", "confidence_score"))

    return predictions


def predict_hate_tweet(df, model_path):
    # load model
    model = LogisticRegressionModel.load(model_path)

    # score the model on test data.
    predictions = model.transform(df)
    predictions = process_prediction(predictions)

    # convert to json
    # {
    #  id: string
    #  tweet: string
    #  tweet_wo_hashtag: string
    #  time_stamp: string
    #  prediction: binary string
    #  confidence_score: double
    # }
    res = list(map(lambda row: row.asDict(), predictions.collect()))

    return res


def push_bad_pred_to_mongo(raw_data, threshold=0.7):
    # select documents with low confidence and push to collection: failed_tweets
    # expected raw data schema:
    # {
    #  id: string
    #  tweet: string
    #  tweet_wo_hashtag: string
    #  time_stamp: string
    #  prediction: binary string
    #  confidence_score: double
    # }
    #
    # return schema:
    # {
    #  id: string
    #  tweet: string
    #  prediction: binary string (hate | not_hate)
    # }

    # MongoDB Atlas connection string
    client = MongoClient("mongodb+srv://admin:admin123@cluster0.agqa7fr.mongodb.net/?retryWrites=true&w=majority")
    db = client.get_database("Tweets")
    
    # get table class
    table = db.failed_tweets
    
    filtered_data = [doc for doc in raw_data if doc['confidence_score']<threshold]
    for doc in filtered_data:
        doc.pop('confidence_score')
        doc.pop('time_stamp')
        doc.pop('tweet_wo_hashtag')
    print(f"==> pushing {len(filtered_data)} bad predictions to DB")
    table.insert_many(filtered_data)


def transform_and_update_mongo(raw_data):
    # expected raw data schema:
    # {
    #  id: string
    #  tweet: string
    #  tweet_wo_hashtag: string
    #  time_stamp: string
    #  prediction: binary string
    #  confidence_score: double
    # }
    #
    # return schema:
    # {
    #  time: DateTime
    #  frequency: list of tuple
    # }
    # MongoDB Atlas connection string
    client = MongoClient("mongodb+srv://admin:admin123@cluster0.agqa7fr.mongodb.net/?retryWrites=true&w=majority")
    db = client.get_database("Tweets")
    
    # push clean tweets
    table = db.clean_tweets
    for sample in raw_data:
        sample["time_stamp"] = datetime.strptime(sample["time_stamp"], "%Y-%m-%dT%H:%M:%S.000Z")
    table.insert_many(raw_data)


    # get table class
    table = db.processed_tweets
    
    data = []
    for query in raw_data:
        time_stamp = query["time_stamp"]
        time_id = str(time_stamp.year)+str(time_stamp.month)+str(time_stamp.day)
        pred = query["prediction"]
        for word in query["tweet_wo_hashtag"].split():
            data.append({
                    "word": word, 
                    "time": time_id, 
                    "prediction": pred
                }
            )
    table.insert_many(data)






c = Consumer(**kafka_conf)
c.subscribe(topics)
data_samples = []
bad_req_count = 0
try:
    while True:
        if bad_req_count > 10:
            break
        if len(data_samples) >= 2000:
            break
        msg = c.poll(timeout=1.0)
        if msg is None:
            print(".", end="")
            sys.stdout.flush()
            bad_req_count += 1
            continue
        if msg.error():
            print(".", end="")
            sys.stdout.flush()
            bad_req_count += 1
            if msg.error().code() == KafkaError._PARTITION_EOF:
                sys.stderr.write('%% %s [%d] reached end at offset %d\n' %
                                    (msg.topic(), msg.partition(), msg.offset()))
            elif msg.error():
                raise KafkaException(msg.error())
        else:
            bad_req_count = 0
            sys.stderr.write('%% %s [%d] at offset %d with key %s:\n' %
                                (msg.topic(), msg.partition(), msg.offset(),
                                str(msg.key())))
            data_samples.append(json.loads(msg.value().decode("utf-8")))
except KeyboardInterrupt:
    sys.stderr.write('%% Aborted by user\n')


# run hate detection inference pipeline
print(f"Total Tweets: {len(data_samples)}")
print("Start Parsing Data")
ids, tweets, time_stamps = get_data(data_samples)
print("Get 100 sample hate tweets for demo")
ids_hate, tweets_hate, time_stamps_hate = get_hate_demo_samples()
ids += ids_hate
tweets += tweets_hate
time_stamps += time_stamps_hate

print("Start Preprocessing Data")
df = preprocess_data(ids, tweets, time_stamps)

print("Transforming Data for ML Model")
df = transform_data(df)

print("Inferencing...")
res = predict_hate_tweet(df, model_path=MODEL_PATH)
pprint(res[:3])

transform_and_update_mongo(res)
print("pushed predicted data to mongoDB")

push_bad_pred_to_mongo(res, threshold=0.7)
print("pushed bad prediction data to mongoDB.")

# Close down consumer to commit final offsets.
c.close()