# BD-HateSpeech
This project implements a system using big data tools to achieve real time hate-speech classification on twitter data stream

<br>

## Requirements for Enviornment
- install spark
- install kafka

<br>

## Initial Model Training
follow [modeling/lr_hate_speech.ipynb](modeling/lr_hate_speech.ipynb) to train the Logistic Regression model

<br>


## Starting the System
```bash
# Run Kafka Producer
python3 TwitterStreaming.py
```
```bash
# Run Kafka Consumer
python3 KafkaConsumer.py
```
```bash
# trigger model retraining
python3 ModelRetraining.py
```

<br>

## Real Time Setup
In real time scenario, you would want to shcedule the `TwitterStreaming` script to run every 15mins so that it periodically fetch data from twitter. You can then schedule the `KafkaConsumer` script to run every 10mins to consume items from the Kafka queue. It is recommended to schedule `ModelRetraining` to update your model every 24 hours but you need to make sure the data in `MongoDB.failed_tweets` are annotated.
