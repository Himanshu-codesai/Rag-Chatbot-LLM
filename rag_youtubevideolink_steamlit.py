from langchain_community.document_loaders import YoutubeLoader 
import io
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import CharacterTextSplitter
import streamlit as st

docs = []

def process_input(urls, question):
    model_local = Ollama(model="mistral")
    
    urls_list = urls.split("\n")
    for url in urls_list:
        loader = YoutubeLoader.from_youtube_url(url)  # Use the current URL in the loop
        docs1 = loader.load()
        for doc in docs1:
            docs.append(doc)
    
    if not docs:
        st.error("No documents loaded from the provided URLs.")
        return

    # Split the text into chunks
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=7500, chunk_overlap=100)
    doc_splits = text_splitter.split_documents(docs)
    
    # Convert text chunks into embeddings and store in vector database
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="rag-chroma",
        embedding=OllamaEmbeddings(model='nomic-embed-text'),
    )
    retriever = vectorstore.as_retriever()
    
    after_rag_template = """Answer the question based only on the following context:
    {context}
    Question: {question}
    """
    after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template)
    after_rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | after_rag_prompt
        | model_local
        | StrOutputParser()
    )
    return after_rag_chain.invoke(question)

st.title("Document Query with Ollama")
st.write("Enter URLs (one per line) and a question to query the documents.")

urls = st.text_area("Enter URLs separated by new lines", height=150)
question = st.text_input("Question")

if st.button('Query Documents'):
    with st.spinner('Processing...'):
        answer = process_input(urls, question)
        st.text_area("Answer", value=answer, height=300, disabled=True)
    