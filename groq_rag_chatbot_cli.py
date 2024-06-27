import os
import PyPDF2
import time  # Import time module for time tracking
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

load_dotenv()

groq_api_key = os.environ.get('GROQ_API_KEY')

llm_groq = ChatGroq(
    groq_api_key=groq_api_key,
    model_name='mixtral-8x7b-32768'
)

def main():
    pdf_path = input("Please enter the path to the PDF file: ")

    if not os.path.exists(pdf_path):
        print(f"File not found: {pdf_path}")
        return

    # Read the PDF file
    pdf = PyPDF2.PdfReader(pdf_path)
    pdf_text = ""
    for page in pdf.pages:
        pdf_text += page.extract_text()

    # Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    texts = text_splitter.split_text(pdf_text)

    # Create metadata for each chunk
    metadatas = [{"source": f"{i}-pl"} for i in range(len(texts))]

    # Create a Chroma vector store
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    docsearch = Chroma.from_texts(texts, embeddings, metadatas=metadatas)

    # Initialize message history for conversation
    message_history = ChatMessageHistory()

    # Memory for conversational context
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )

    # Create a chain that uses the Chroma vector store
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm_groq,
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        memory=memory,
        return_source_documents=True,
    )

    print(f"Processing `{pdf_path}` done. You can now ask questions!")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        # Track time before making the chain call
        start_time = time.time()

        # Call the chain with the user's input
        res = chain(user_input)
        answer = res["answer"]
        source_documents = res["source_documents"]

        # Calculate time taken
        end_time = time.time()
        time_taken = end_time - start_time

        if source_documents:
            source_names = [f"source_{i}" for i, _ in enumerate(source_documents)]
            answer += f"\nSources: {', '.join(source_names)}"

        # Print answer and time taken
        print(f"AI: {answer}")
        print(f"Time taken: {time_taken:.2f} seconds")

if __name__ == "__main__":
    main()
