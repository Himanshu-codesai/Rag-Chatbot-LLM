import os
import psycopg2
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
import time

llm = Ollama(model="mistral")

def fetch_pdf_from_db(filename):
    try:
        connection = psycopg2.connect(
            dbname="ragpdf",
            user="postgres",
            password="123",
            host="localhost",
            port="5432"
        )
        cursor = connection.cursor()
        query = "SELECT file_data FROM pdf_files WHERE filename = %s"
        cursor.execute(query, (filename,))
        result = cursor.fetchone()
        if result:
            with open(filename, 'wb') as file:
                file.write(result[0])
            return filename
        return None
    except Exception as e:
        print(f"Error fetching PDF: {e}")
    finally:
        if connection:
            cursor.close()
            connection.close()

def load_chunk_pdf(pdf_files):
    documents = []
    
    for pdf_file in pdf_files:
        loader = PyPDFLoader(pdf_file)
        documents.extend(loader.load())
    
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunked_documents = text_splitter.split_documents(documents)
    
    return chunked_documents

def main():
    question_start_time = time.time()

    print("PDF Document Chunking and Analysis")

    pdf_filename = "test.pdf"
    fetched_pdf = fetch_pdf_from_db(pdf_filename)
    
    if fetched_pdf:
        pdf_files = [fetched_pdf]
        chunked_documents = load_chunk_pdf(pdf_files)

        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=1500, chunk_overlap=200)
        doc_splits = text_splitter.split_documents(chunked_documents)

        vectorstore = Chroma.from_documents(
            documents=doc_splits,
            collection_name="rag-chroma",
            embedding=OllamaEmbeddings(model='nomic-embed-text'),
        )

        retriever = vectorstore.as_retriever()

        PROMPT_TEMPLATE = """
        Answer the question based only on the following context:
        {context}
        Answer the question based on the above context: {question}.
        """
        
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

        print("PDF documents processed successfully!")

        while True:
            user_question = input("\nEnter your question (or 'exit' to quit): ")
            if user_question.lower() == 'exit':
                break
            
            if len(doc_splits) > 0:
                context = doc_splits[0].page_content    # Using the first chunk as context
                prompt = prompt_template.format(context=context, question=user_question)
                response = llm.generate([prompt])
                
                answer = response.generations[0][0].text
                
                question_end_time = time.time()
                
                print(f"\nAnswer: {answer}")
                print(f"Time taken to answer the question: {question_end_time - question_start_time:.2f} seconds")
            else:
                print("No document chunks available.")
    else:
        print("Failed to fetch PDF from the database.")

if __name__ == "__main__":
    main()
