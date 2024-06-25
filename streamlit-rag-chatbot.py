import os
import time
import tempfile
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate

# Initialize the LLM
llm = Ollama(model="mistral")

# Function to load and chunk PDF files
def load_chunk_pdf(pdf_file_path):
    documents = []
    loader = PyPDFLoader(pdf_file_path)
    documents.extend(loader.load())
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunked_documents = text_splitter.split_documents(documents)
    return chunked_documents

# Main function to process the uploaded PDF and answer questions
def main():
    st.title("Document Query with Ollama")
    st.write("Upload a PDF document and enter a question to query the document.")

    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    question = st.text_input("Enter your question")

    if st.button('Query Document'):
        if uploaded_file is not None and question:
            with st.spinner('Processing...'):
                start_time = time.time()

                # Use a temporary file to handle the uploaded file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                    temp_file.write(uploaded_file.read())
                    temp_filepath = temp_file.name

                # Load and chunk the PDF
                chunked_documents = load_chunk_pdf(temp_filepath)
                text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=1500, chunk_overlap=200)
                doc_splits = text_splitter.split_documents(chunked_documents)

                # Create the vector store
                vectorstore = Chroma.from_documents(
                    documents=doc_splits,
                    collection_name="rag-chroma",
                    embedding=OllamaEmbeddings(model='nomic-embed-text'),
                )

                # Initialize retriever
                retriever = vectorstore.as_retriever()

                PROMPT_TEMPLATE = """
                Answer the question based only on the following context:
                {context}
                Answer the question based on the above context: {question}.
                """
                
                prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

                if len(doc_splits) > 0:
                    context = doc_splits[0].page_content  # Using the first chunk as context
                    prompt = prompt_template.format(context=context, question=question)
                    response = llm.generate([prompt])
                    answer = response.generations[0][0].text

                    end_time = time.time()
                    time_taken = end_time - start_time

                    st.text_area("Answer", value=answer, height=300, disabled=True)
                    st.write(f"Time taken to answer the question: {time_taken:.2f} seconds")
                else:
                    st.write("No document chunks available.")
                
                # Remove the temporary file
                os.remove(temp_filepath)
        else:
            st.write("Please upload a PDF file and enter a question.")

if __name__ == "__main__":
    main()
