from transformers import pipeline
import pandas as pd
import csv

data = pd.read_csv("data.csv")

notes = data['PNT_ATRISKNOTES_TX'].to_list()

# print(notes)

# llm = pipeline("text-2-text")

# print(llm(notes[0:5]))

from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")

results = [["Output", "Note"]]

for i in range(0, len(data) // 100):
    input_text = "classify the following situation as dangerous or not dangerous: " + notes[i]
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    outputs = model.generate(input_ids)
    # print(tokenizer.decode(outputs[0]))
    results.append(
        [tokenizer.decode(outputs[0]), notes[i]]
        )

with open('results.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(results)