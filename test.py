import torch
from pandas import *
import pandas as pd
import requests
from transformers import AutoTokenizer, DistilBertForSequenceClassification, GPT2Tokenizer, GPT2ForSequenceClassification, AutoModelForCasualLM
from transformers import pipeline
import torch.optim as optim

# reading CSV file
data = read_csv("aep_data.csv")
# get a column of comments
comments = data['PNT_ATRISKNOTES_TX'].tolist()
#print('Comments: ', comments)


model_path = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
#sentiment_task = pipeline("sentiment-analysis")
#sentiment_task("Hello!")

# Load the pre-trained tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Load the pre-trained model for sequence classification
model = GPT2ForSequenceClassification.from_pretrained('gpt2')

texts = []
i = 0
for comment in comments:
    texts.append(comment)
    if i > 4:
        break
    i = i + 1

sentiments = ["positive", "negative", "neutral"]
instructions = ["Analyze the sentiment of the text and identify if it is positive.",
                "Analyze the sentiment of the text and identify if it is negative.",
                "Analyze the sentiment of the text and identify if it is neutral."]


# Tokenize the texts, sentiments, and instructions
encoded_texts = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
encoded_instructions = tokenizer(instructions, padding=True, truncation=True, return_tensors='pt')

# Extract input IDs, attention masks, and instruction IDs
input_ids = encoded_texts['input_ids']
attention_mask = encoded_texts['attention_mask']
instruction_ids = encoded_instructions['input_ids']

# Concatenate instruction IDs with input IDs and adjust attention mask
input_ids = torch.cat([instruction_ids, input_ids], dim=1)
attention_mask = torch.cat([torch.ones_like(instruction_ids), attention_mask], dim=1)




# part from earlier
#Tokenizer = AutoTokenizer.from_pretrained(MODEL)
limit = 5
for comment in comments:
    limit = limit - 1
    if (limit >= 0):
        #sentiment_task = pipeline("sentiment-analysis", model=model_path, tokenizer=model_path)
        sentiment_task = pipeline("sentiment-analysis")
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