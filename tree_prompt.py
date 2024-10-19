import pandas as pd
import csv
from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")

data = pd.read_csv("data.csv")

log = open("log.txt", "w")

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

# this prompt is the priority.
def high_energy_present(note):
    question = "Does the following scenario have any dangerous hazards?"
    # question = "Does the following scenario have any of the following hazards: a suspended load, a person above 4 feet off the ground, moving equipment near people, a vehicle moving faster than 30 miles per hour, heavy rotating equipment, temperature greater than 150 degrees fahrenheit, fire, explosion, exposure to unsupported soil more than 5 feet deep, electricity exceeding 50 volts or an arc flash, exposure to toxic chemicals or radiation?  " # TODO
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
    response = tokenizer.decode(model.generate(input_ids)[0])
    log.write(f'Prompt: {prompt}\n Response: {response} \n')
    return response

def yn_to_tf(output):
    if "yes" in output:
        return True
    else:
        return False

for note in notes[0:100]:
    print(high_energy_present(note))

print("done")
log.close()