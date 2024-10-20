import torch
import transformers
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import pandas as pd

'''
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

aep_dataset = pd.read_csv(
    "C:\\Users\\Black\\Downloads\\CORE_HackOhio_subset_cleaned_downsampled 1.csv"
)

# pipeline = pipeline("text2text-generation", model=model, device=-1, tokenizer=tokenizer, max_length=1000)

classifier = transformers.pipeline(
    task="text-classification",
    model=model,
    tokenizer=tokenizer,
    max_length=100,
)  # Note: We specify cache_dir to use predownloaded models.

results = classifier(aep_dataset["PNT_ATRISKNOTES_TX"].to_list()[0:5])
print(results)
'''
'''
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor


Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

Settings.llm
Settings.chunk_size = 256
Settings.chunk_overlap = 25

documents = SimpleDirectoryReader("Data").load_data()
index = VectorStoreIndex.from_documents(documents)

print(documents)


# Setting up a retriever
top = 3

retriever = VectorIndexRetriever(
    index=index,
    similarity_top = top
)

# Assemble query engine
query_engine = RetrieverQueryEngine(
    retriever=retriever,
    node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.5)],
)

query = "What are high value indicators?"
response = query_engine.query(query)

# reformat response
context = "Context: \n"
for i in range(top):
    context = context + response.sources_nodes[i].text + "\n\n"

print(context)
'''

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import DirectoryLoader
loader = DirectoryLoader("Data", glob="**/*.txt")
data = loader.load()
# print(len(data))

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

local_embeddings = OllamaEmbeddings(model="gemma2:2b")

vectorstore = Chroma.from_documents(documents=all_splits, embedding=local_embeddings)

# question = "What is Capacity?"
# docs = vectorstore.similarity_search(question)
# print(len(docs))
# print(docs[0])

# Set the model 

from langchain_ollama import ChatOllama
model = ChatOllama(
    model="gemma2:2b",
    max_tokens=1
)

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template(
    "Given the following information on the presence of high-energy: {docs}, does the following scenario clearly have any high-energy elements, yes or no? {scenario}"
)

# Convert loaded documents into strings by concatenating their content
# and ignoring metadata
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# def scenario_string(test):
#     return "An employee was on the top of a de-energized transformer at 25 feet of height with a proper fall arrest system. While working, she tripped on a lifting lug, falling within 2 feet from an unguarded edge. When the employee landed, she sprained her wrist."

# def scenario_string(test):
#     return "An employee contracted West Nile Virus after being bitten by a mosquito while at work in a boggy area. Because of the exposure, the employee was unconscious and paralyzed for a two-week period."

# def scenario_string(test):
#     return "An employee was working alone and placed an extension ladder against the wall. When he reached 10 feet of height, the ladder feet slid out and he fell with the ladder to the floor. The employee was taken to the hospital for a bruise to his right leg and remained off duty for three days."

# def scenario_string(test):
#     return "A crew was closing a 7-ton door on a coal crusher. As the door was lowered, an observer noticed that the jack was not positioned correctly and could tip. The observer also noted that workers were nearby, within 4 feet of the jack."

# def scenario_string(test):
#     return "Workers were hoisting beams and steel onto a scaffold. A certified mechanic operated an air hoist to lift the beam. As the lift was performed, the rigging was caught under an adjacent beam. Under the increasing tension, the cable snapped and struck a second employee in the leg, fully fracturing his femur. An investigation indicated that the rigging was not properly inspected before the lift."

# def scenario_string(test):
#     return "A dozer was operating on a pet coke pile and slid down an embankment onto the cab after encountering a void in the pile. The operator was wearing his seat belt, and the roll cage kept the cab from crushing. No workers or machinery were nearby, and no injuries were sustained."

# def scenario_string(test):
#     return "A master electrician was called to work on a new 480-volt service line in a commercial building. When working on the meter cabinet, the master electrician had to position himself awkwardly between the cabinet and a standpipe. He was not wearing an arc-rated face shield, balaclava, or proper gloves. During the work, an arc flash occurred, causing third-degree burns to his face."

# this one is 50/50 for gemma
# def scenario_string(test):
#     return "An employee was descending a staircase and when stepping down from the last step she rolled her ankle on an extension cord on the floor. She suffered a torn ligament and a broken ankle that resulted in persistent pain for more than a year."

# def scenario_string(test):
#     return "A crew was working near a sedimentation pond on a rainy day. The boom of the trac-hoe was within 3 feet of a live 12kV line running across the road. No contact was made because a worker intervened and communicated with the operator."

# def scenario_string(test):
#      return "A crew was working in a busy street to repair a cable fault. During the work, the journeyman took a step back from the truck outside of the protected work zone into oncoming traffic. A driver slammed on his brakes and stopped within one foot of the journeyman. No injuries were sustained."

chain = {"docs": format_docs, "scenario": scenario_string} | prompt | model | StrOutputParser()

question = "What scenarios are high energy"

docs = vectorstore.similarity_search(question)

print(chain.invoke(docs))

'''
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

local_embeddings = OllamaEmbeddings(model="nomic-embed-text")

vectorstore = Chroma.from_documents(documents=all_splits, embedding=local_embeddings)

question = "What are the approaches to Task Decomposition?"
docs = vectorstore.similarity_search(question)
print(len(docs))

from langchain_ollama import ChatOllama

model = ChatOllama(
    model="mistral",
)

response_message = model.invoke(
    "Simulate a rap battle between Stephen Colbert and John Oliver"
)

print(response_message.content)
'''
