import pandas as pd
import csv
from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")

data = pd.read_csv("data.csv")

notes = data['PNT_ATRISKNOTES_TX'].to_list()

def run_decision_tree(note):
    if high_energy_present(note):
        if high_energy_incident(note):
            if serious_injury(note):
                return "HSIF"
            else:
                if direct_control(note):
                    return "Capacity"
                else:
                    return "PSIF"
        else:
            if direct_control(note):
                return "Success"
            else:
                return "Exposure"
    else: 
        if serious_injury(note):
            return "LSIF"
        else:
            return "Low Severity"

def serious_injury(note):
    question = "Did a serious injury occur in the following scenario?"
    response = askLLM(generateYNPrompt(note, question))
    return yn_to_tf(response)

def direct_control(note):
    question = "" # TODO
    response = askLLM(generateYNPrompt(note, question))
    return yn_to_tf(response)

def high_energy_present(note):
    question = "" # TODO
    response = askLLM(generateYNPrompt(note, question))
    return yn_to_tf(response)

def high_energy_incident(note):
    question = "" # TODO
    response = askLLM(generateYNPrompt(note, question))
    return yn_to_tf(response)

def generateYNPrompt(note, question):
    prompt = question + " Answer with yes or no.  The scenario is: " + note
    return prompt

def askLLM(prompt):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    return model.generate(input_ids)

def yn_to_tf(output):
    if "yes" in str(output):
        return True
    else:
        return False


for note in notes[0:100]:
    print(serious_injury(notes[0]))

print("done")