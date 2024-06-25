from flask import Flask, request, jsonify
import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate

app = Flask(__name__)
llm = Ollama(model="mistral")

doc_splits = []
prompt_template = None

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def load_chunk_pdf(pdf_files):
    documents = []
    try:
        for pdf_file in pdf_files:
            temp_filepath = os.path.join(app.config['UPLOAD_FOLDER'], pdf_file)
            print(f"Loading PDF file from: {temp_filepath}")  # Debug statement
            loader = PyPDFLoader(temp_filepath)
            loaded_docs = loader.load()
            if not loaded_docs:
                print(f"Failed to load documents from: {pdf_file}")  # Debug statement
            documents.extend(loaded_docs)
        
        if not documents:
            print("No documents loaded from PDFs.")  # Debug statement

        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunked_documents = text_splitter.split_documents(documents)
        
        if not chunked_documents:
            print("No document chunks created.")  # Debug statement

        return chunked_documents
    
    except Exception as e:
        print(f"Error in load_chunk_pdf: {str(e)}")  # Debug statement
        return []

@app.route('/upload', methods=['POST'])
def process_pdf():
    global doc_splits, prompt_template
    try:
        print(f"Request files: {request.files}")  # Debug statement
        if 'pdf_files' not in request.files:
            return jsonify({"error": "No files uploaded."}), 400
        
        files = request.files.getlist('pdf_files')
        pdf_files = []
        
        # Save uploaded files to the server
        for file in files:
            if file.filename == '':
                return jsonify({"error": "No selected file."}), 400
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            pdf_files.append(file.filename)
            print(f"Saved file to: {file_path}")  # Debug statement
        
        chunked_documents = load_chunk_pdf(pdf_files)
        if not chunked_documents:
            return jsonify({"error": "No document chunks found."}), 500

        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=1500, chunk_overlap=200)
        doc_splits = text_splitter.split_documents(chunked_documents)
        
        if not doc_splits:
            return jsonify({"error": "Document splitting failed."}), 500

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

        return jsonify({"message": "PDF documents processed successfully!"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/query', methods=['POST'])
def ask_question():
    global doc_splits, prompt_template
    try:
        data = request.get_json()
        question = data['question']
        
        if len(doc_splits) > 0:
            context = doc_splits[0].page_content    # Using the first chunk as context
            prompt = prompt_template.format(context=context, question=question)
            response = llm.generate([prompt])
            answer = response.generations[0][0].text
            
            return jsonify({"answer": answer})
        else:
            return jsonify({"error": "No document chunks available."}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
