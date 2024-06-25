import os
import sys
# print(os.path.dirname(sys.executable))
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
import time

llm = Ollama(model="mistral")


def load_chunk_pdf(pdf_files):
    documents = []
    
    for pdf_file in pdf_files:
        temp_filepath = os.path.join(os.path.dirname(__file__), pdf_file)
        
        loader = PyPDFLoader(temp_filepath)
        documents.extend(loader.load())
    
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunked_documents = text_splitter.split_documents(documents)
    
    return chunked_documents


def main():
    start_time = time.time()

    print("PDF Document Chunking and Analysis")

    pdf_files = ["test.pdf"]  # Example: list PDF file paths here
    
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
            
            print(f"\nAnswer: {answer}")
        else:
            print("No document chunks available.")
            
if __name__ == "__main__":
    main()
