import torch
from pandas import *
import pandas as pd
import requests
from transformers import AutoTokenizer, DistilBertForSequenceClassification
from transformers import pipeline

# reading CSV file
data = read_csv("aep_data.csv")
# get a column of comments
comments = data['PNT_ATRISKNOTES_TX'].tolist()
#print('Comments: ', comments)


model_path = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
#sentiment_task = pipeline("sentiment-analysis")
#sentiment_task("Hello!")

Tokenizer = AutoTokenizer.from_pretrained(MODEL)
limit = 5
for comment in comments:
    limit = limit - 1
    if (limit >= 0):
        sentiment_task = pipeline("sentiment-analysis", model=model_path, tokenizer=model_path)
        #sentiment_task = pipeline("sentiment-analysis")
        #sentiment_task(comment)
        print(sentiment_task(comment))
        #sentiment_task(comment)
        #print(f"{comment}: {sentiment_task}")

"""
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")


inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits

predicted_class_id = logits.argmax().item()
model.config.id2label[predicted_class_id]

"""