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

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import DirectoryLoader
loader = DirectoryLoader("Data", glob="**/*.txt")
data = loader.load()
print(len(data))

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

local_embeddings = OllamaEmbeddings(model="mistral")

vectorstore = Chroma.from_documents(documents=all_splits, embedding=local_embeddings)

question = "What is Capacity?"
docs = vectorstore.similarity_search(question)
print(len(docs))
print(docs[0])

# Set the model 

from langchain_ollama import ChatOllama
model = ChatOllama(
    model="mistral",
)

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template(
    "Summarize the main themes in these retrieved docs: {docs}"
)

# Convert loaded documents into strings by concatenating their content
# and ignoring metadata
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


chain = {"docs": format_docs} | prompt | model | StrOutputParser()

question = "What is Capacity?"

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
