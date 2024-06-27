# Rag-Chatbot-LLM

## Installation

### Create Virtual Environment
```
pip install virtualenv
virtualenv venv
venv\Scripts\activate
```
### Clone the Repository
```
Git clone https://github.com/Himanshu-codesai/Rag-Chatbot-LLM.git
```
### Install Dependencies
```
pip install -r requirements.txt
```
### Set Up Ollama
- We use Ollama to run the llms model on our local server.
[Download Ollama from here](https://ollama.com/download)

- After Ollama installation , open cmd terminal and put below two commands - 
```
Ollama pull mistral & Ollama pull nomic-embed-text
```
- For more next detailed information, please refer to the [Report.docx](Report.docx).

### Optional Codes
- I have also implemented some code using streamlit,Groq based implementation of RAG. 
