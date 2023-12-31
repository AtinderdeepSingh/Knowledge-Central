import os
import streamlit as st
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.azuresearch import AzureSearch
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import pandas as pd

load_dotenv()

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]                                           #OpenAI API key
vector_store_address = st.secrets["vector_store_address"]                               #Azure AI search endpoint
vector_store_password = st.secrets["vector_store_password"]                             #Azure AI search key
account_name = st.secrets["account_name"]                                               #Azure blob storage account name
constring = st.secrets["constring"]                                                     #Azure blob storage connection string

aoai_embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo')

st.title("Knowledge Center 📚")
st.sidebar.title("Select Data Source")
option = st.sidebar.selectbox(
    'Available Vector Indexes:',
    ('Retriever1', 'Retriever2', 'Retriever3', 'Retriever4'))
st.sidebar.write("You are querying :", option)

if 'Retriever1' in option:
    index_name = "cleanindex"
    vectordb = AzureSearch(azure_search_endpoint=vector_store_address, azure_search_key=vector_store_password, index_name=index_name, embedding_function=aoai_embeddings.embed_query)
    custom_prompt = """Given the following context and a question, generate an answer based on this context only. If the answer is not found in the context, kindly state "I cannot find answer in the given vector store" Don't try to make up an answer.
    CONTEXT: {context}
    QUESTION: {question}
    """
    custom_prompt_template = PromptTemplate(template=custom_prompt, input_variables=["context", "question"])
    retriever = vectordb.as_retriever(search_type="similarity",search_kwargs={"k": 3})
if 'Retriever4' in option:
    index_name = "cleanindex"
    vectordb = AzureSearch(azure_search_endpoint=vector_store_address, azure_search_key=vector_store_password, index_name=index_name, embedding_function=aoai_embeddings.embed_query)
    custom_prompt = """Use your pre trained knowledge to answer the question.
    CONTEXT: {context}
    QUESTION: {question}
    """
    custom_prompt_template = PromptTemplate(template=custom_prompt, input_variables=["context", "question"])
    retriever = vectordb.as_retriever(search_type="similarity",search_kwargs={"k": 3})
else:
    index_name = None

# Dataframe to store results for evaluation - Prompts, context retrieved, generated answer
df=pd.DataFrame()
questions=[]                        #Initializing an empty question list
context=[]                          #Initializing an empty context list
answer=[]                           #Initializing an empty answer list

main_placeholder = st.empty()
query = main_placeholder.text_input("Question: ")
if query: 
    questions.append(query)   
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True, chain_type_kwargs={"prompt": custom_prompt_template})
    result = qa_chain(query)
    st.header("Answer")
    st.write(result["result"])
    context_all=[]
    for i in range(3):
        context_all.append(result['source_documents'][i].page_content)
    context.append(context_all)
    answer.append(result["result"])

    # Display sources   
    sources=[]
    for i in range(3):
        sources.append(result["source_documents"][i].metadata['source'])
    if sources:
        st.subheader("Sources:")
        for source in sources:
            st.write(source)
    
    df['question']=questions
    df['answer']=answer
    df['context']=context
    df.to_csv('results.csv', mode='a', header=False)
