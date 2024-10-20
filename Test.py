import pandas as pd
import ssl
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama

def standardize_res(res):
    res = res.lower()
    if "yes" in res:
        return "yes"
    elif "no" in res:
        return "no"
    else:
        return "unsure"


# Read csv data and get only the notes.
input_csv = pd.read_csv("data.csv")
notes = input_csv["PNT_ATRISKNOTES_TX"].to_list()

# load and prepare context data
ssl._create_default_https_context = ssl._create_unverified_context
loader = DirectoryLoader("Data", glob="**/*.txt")
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)

local_embeddings = OllamaEmbeddings(model="gemma2:2b")

vectorstore = Chroma.from_documents(documents=all_splits, embedding=local_embeddings)

# Set the model 
model = ChatOllama(
    model="gemma2:2b",
    max_tokens=1
)

# create a prompt template
prompt = ChatPromptTemplate.from_template(
    "Given the following information on the presence of high-energy: {docs}, does the following scenario clearly have any high-energy elements, yes or no? Do not explain your answer. {scenario}"
)

# Convert loaded documents into strings by concatenating their content
# and ignoring metadata
def format_docs(docsAndNotes):
    docs, notes = docsAndNotes
    return "\n\n".join(doc.page_content for doc in docs)

# Get next scenario string from list
# not sure if this is the best way to do this
def scenario_string(docsAndNotes):
    docs, note = docsAndNotes
    return note

# Yes
# def scenario_string(test):
#     return "An employee was on the top of a de-energized transformer at 25 feet of height with a proper fall arrest system. While working, she tripped on a lifting lug, falling within 2 feet from an unguarded edge. When the employee landed, she sprained her wrist."

# No
# def scenario_string(test):
#     return "An employee contracted West Nile Virus after being bitten by a mosquito while at work in a boggy area. Because of the exposure, the employee was unconscious and paralyzed for a two-week period."

# Yes
# def scenario_string(test):
#     return "An employee was working alone and placed an extension ladder against the wall. When he reached 10 feet of height, the ladder feet slid out and he fell with the ladder to the floor. The employee was taken to the hospital for a bruise to his right leg and remained off duty for three days."

# Yes
# def scenario_string(test):
#     return "A crew was closing a 7-ton door on a coal crusher. As the door was lowered, an observer noticed that the jack was not positioned correctly and could tip. The observer also noted that workers were nearby, within 4 feet of the jack."

# Yes
# def scenario_string(test):
#     return "Workers were hoisting beams and steel onto a scaffold. A certified mechanic operated an air hoist to lift the beam. As the lift was performed, the rigging was caught under an adjacent beam. Under the increasing tension, the cable snapped and struck a second employee in the leg, fully fracturing his femur. An investigation indicated that the rigging was not properly inspected before the lift."

# Yes
# def scenario_string(test):
#     return "A dozer was operating on a pet coke pile and slid down an embankment onto the cab after encountering a void in the pile. The operator was wearing his seat belt, and the roll cage kept the cab from crushing. No workers or machinery were nearby, and no injuries were sustained."

# Yes
# def scenario_string(test):
#     return "A master electrician was called to work on a new 480-volt service line in a commercial building. When working on the meter cabinet, the master electrician had to position himself awkwardly between the cabinet and a standpipe. He was not wearing an arc-rated face shield, balaclava, or proper gloves. During the work, an arc flash occurred, causing third-degree burns to his face."

# this one may be less consistent for gemma
# No
# def scenario_string(test):
#     return "An employee was descending a staircase and when stepping down from the last step she rolled her ankle on an extension cord on the floor. She suffered a torn ligament and a broken ankle that resulted in persistent pain for more than a year."

# Yes
# def scenario_string(test):
#     return "A crew was working near a sedimentation pond on a rainy day. The boom of the trac-hoe was within 3 feet of a live 12kV line running across the road. No contact was made because a worker intervened and communicated with the operator."

# Yes
# def scenario_string(test):
#      return "A crew was working in a busy street to repair a cable fault. During the work, the journeyman took a step back from the truck outside of the protected work zone into oncoming traffic. A driver slammed on his brakes and stopped within one foot of the journeyman. No injuries were sustained."

chain = {"docs": format_docs, "scenario": scenario_string} | prompt | model | StrOutputParser()

# Narrow down context 
question = "What scenarios are high energy"
docs = vectorstore.similarity_search(question)

log = open("log.txt", "w")
# while notes is not empty
col = []
count = 0
for note in notes:
    res = chain.invoke((docs, note))
    standard_res = standardize_res(res)
    col.append(standard_res)
    count += 1
    if count % 10 == 0:
        print(count)

input_csv.insert(6, "High_Value", col, True)
input_csv.to_csv('labelled_data.csv', index=False)
