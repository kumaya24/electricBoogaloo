import pandas as pd
import ssl
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama

#standardize the response from gemma
def standardize_res(res):
    res = res.lower()
    if "yes" in res:
        return "yes"
    elif "no" in res:
        return "no"
    else:
        return "unsure"

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

def runModelOnCSV(filename):
    # Read csv data and get only the notes.
    input_csv = pd.read_csv(filename)
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

    chain = {"docs": format_docs, "scenario": scenario_string} | prompt | model | StrOutputParser()

    # Narrow down context 
    question = "What scenarios are high energy"
    docs = vectorstore.similarity_search(question)

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
    # clear output file
    output = "labeled_data.csv"
    open(output, "w").close()

    # write to output file
    input_csv.to_csv(output, index=False)
