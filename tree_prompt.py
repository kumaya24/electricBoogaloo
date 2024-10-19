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
    # TODO
    return

def direct_control(note):
    # TODO
    return

def high_energy_present(note):
    # TODO
    return

def high_energy_incident(note):
    # TODO
    return


