import os
import langchain
from dotenv import load_dotenv
from langchain.document_loaders import AzureBlobStorageContainerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.azuresearch import AzureSearch

load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")                                           #OpenAI API key
vector_store_address = os.environ.get("vector_store_address")                               #Azure AI search endpoint
vector_store_password = os.environ.get("vector_store_password")                             #Azure AI search key
account_name = os.environ.get("account_name")                                               #Azure blob storage account name
constring = os.environ.get("constring")                                                     #Azure blob storage connection string

con = "test-data"                                                                           #Container name that contains the data to be indexed


#Preparing loader to load data from the blob storage as document. A Document is a piece of text and associated metadata
loader = AzureBlobStorageContainerLoader( conn_str=constring, container=con)                #Configuring loader
documents=loader.load()                                                                     #Loading data as document

#Splitting the text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)          #Configuring splitter
texts = text_splitter.split_documents(documents)                                            #Running splitter
print("Number of chunks- ",len(texts))

aoai_embeddings = OpenAIEmbeddings()
index_name: str = "cleanindex"
vectordb: AzureSearch = AzureSearch( azure_search_endpoint=vector_store_address, azure_search_key=vector_store_password, index_name=index_name, embedding_function=aoai_embeddings.embed_query, search_type="hybrid")

#Pushing chunks into vector store
vectordb.add_documents(documents=texts)
print("Index is created...")
