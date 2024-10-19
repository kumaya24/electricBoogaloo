from transformers import GPT2Tokenizer

# Loading the dataset to train our model
# reading CSV file
dataset = read_csv("aep_data.csv")
# get a column of comments
texts = dataset['PNT_ATRISKNOTES_TX'].tolist()
#texts = comments[0:5]

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
def tokenize_function(examples):
   return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)