from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2ForSequenceClassification, TrainingArguments, AutoModelForCausalLM, AutoTokenizer
import torch
import torch.optim as optim
from pandas import *
import pandas as pd
import torch.nn as nn

# Load the pre-trained tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
#tokenizer = AutoTokenizer.from_pretrained("bigcode/octocoder")
tokenizer.pad_token = tokenizer.eos_token

#training_args = TrainingArguments(per_device_train_batch_size = 1, per_device_eval_batch_size = 1)


# reading CSV file
data = read_csv("aep_data.csv")
# get a column of comments
texts = data['PNT_ATRISKNOTES_TX'].tolist()


"""STEP 2"""
# Load the pre-trained model for sequence classification
#model = GPT2ForSequenceClassification.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained("distilgpt2", low_cpu_mem_usage=True)
#model = AutoModelForCausalLM.from_pretrained("bigcode/octocoder", torch_dtype=torch.bfloat16, device_map="auto", pad_token_id=0)

#trainer = Trainer(model=model, args=training_args, train_dataset=data)

sentiments = ["dangerous", "safe"]
"""instructions = ["Analyze the sentiment of the text and identify if it is dangerous.",
                "Analyze the sentiment of the text and identify if it is safe."]"""

encoded_texts = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

# Extract the input IDs and attention masks
input_ids = encoded_texts['input_ids']
attention_mask = encoded_texts['attention_mask']

# Convert the sentiment labels to numerical form
sentiment_labels = [sentiments.index(sentiment) for sentiment in sentiments]

# Add a custom classification head on top of the pre-trained model
num_classes = len(set(sentiment_labels))
classification_head = nn.Linear(model.config.hidden_size, num_classes)

# Replace the pre-trained model's classification head with our custom head
model.classifier = classification_head

# # Tokenize the texts, sentiments, and instructions
# encoded_texts = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
# encoded_instructions = tokenizer(instructions, padding=True, truncation=True, return_tensors='pt')

# # Extract input IDs, attention masks, and instruction IDs
# input_ids = encoded_texts['input_ids']
# attention_mask = encoded_texts['attention_mask']
# instruction_ids = encoded_instructions['input_ids']

# # Concatenate instruction IDs with input IDs and adjust attention mask
# input_ids = torch.cat([instruction_ids, input_ids], dim=1)
# attention_mask = torch.cat([torch.ones_like(instruction_ids), attention_mask], dim=1)

# Define the optimizer and loss function
optimizer = optim.AdamW(model.parameters(), lr=2e-5)
criterion = torch.nn.CrossEntropyLoss()

# Fine-tune the model
num_epochs = 3
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(input_ids, attention_mask=attention_mask, labels=torch.tensor(sentiment_labels))
    loss = outputs.loss
    loss.backward()
    optimizer.step()