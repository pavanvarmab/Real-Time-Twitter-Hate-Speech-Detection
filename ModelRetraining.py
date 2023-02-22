import sys
import os
import json

from dotenv import load_dotenv
load_dotenv()

from nltk.corpus import stopwords
import nltk
from typing import *

import re
import pyspark
from pyspark.sql.functions import udf, when
from pyspark.sql.types import StringType
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml.classification import LogisticRegression

from pymongo import MongoClient
import random


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


def get_data(data_samples):
    ids = []
    tweets = []
    labels = []
    for data in data_samples:
        ids.append(data["id"])
        tweets.append(data["tweet"])

        # since we dont have really human annotator
        # we will flip the model prediction and use it as true label
        label = 0
        if random.random() >= 0.5:
            label = 1
        labels.append(label)
    
    return ids, tweets, labels


def get_tweets_for_retrain() -> List[str]:
    # MongoDB Atlas connection string
    client = MongoClient("mongodb+srv://admin:admin123@cluster0.agqa7fr.mongodb.net/?retryWrites=true&w=majority")
    db = client.get_database("Tweets")
    
    # get data
    table = db.failed_tweets
    data = list(table.find({}))
    ids, tweets, labels = get_data(data)
    
    return ids, tweets, labels


def preprocess_data(ids: List[str], tweets: List[str], labels: List[int]):
    # create pyspark df
    column_names = ["id", "tweet", "label"]
    data = list(zip(ids, tweets, labels))
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

    # since there are class imbalance, add class weight
    hate_df = rescaledData.filter("label == 1")
    hate_ratio = hate_df.count() / rescaledData.count()
    hate_class_weight = 1 - hate_ratio
    neg_class_weight = hate_ratio

    # add weight column
    rescaledData = rescaledData.withColumn(
        "weight", 
        when(rescaledData.label == 1, hate_class_weight).otherwise(neg_class_weight)
    )

    return rescaledData


def train_model(train, maxIter=1000, regParam=1.0, family="auto", elasticNetParam=0,
                weightCol=None, output_path="ouputs/", store_model=True):
    # instantiate the base classifier.
    lr = LogisticRegression(maxIter=maxIter, tol=1E-6, regParam=regParam, 
                            weightCol=weightCol, family=family, elasticNetParam=elasticNetParam)

    # train the multiclass model.
    lrModel = lr.fit(train)
    
    # store model checkpoint
    if store_model:
        lrModel.write().overwrite().save(output_path)
        print("model stored!")


def retrain_model():
    # prepare train test set
    print("Get data for retraining")
    ids, tweets, labels = get_tweets_for_retrain()
    print("Total samples:", len(ids))
    
    print("Preprocess and Transform data for model training")
    df = preprocess_data(ids, tweets, labels)
    rescaledData = transform_data(df)

    print("Train model")
    # best-model hyperparameters
    maxIter = 1000
    regParam = 0.1
    weightCol = "weight"
    family = "binomial"
    elasticNetParam = 0.015
    output_path = "lr_bestModel/"

    # train model
    train_model(rescaledData, maxIter, 
                regParam=regParam, weightCol=weightCol, 
                family=family, elasticNetParam=elasticNetParam,
                output_path=output_path, store_model=True)




retrain_model()





