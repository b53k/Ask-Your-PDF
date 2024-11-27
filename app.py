import os
import tiktoken
import pdfplumber
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv


# Initialize Flask application
app = Flask(__name__)
CORS(app)

# Load environment variables form a .env file
load_dotenv()

# API Key Setup
api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = api_key

# Initialize the LLM and settings
llm = ChatOpenAI(model = "gpt-4o")
TOKEN_LIMIT = 4000
conversation_history = []
vector_store = None

def count_tokens(text, model="gpt-4o"):
    """Estimate the number of tokens in a given text."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


def truncate_history(conversation_history, max_tokens):
    """Truncate conversation history to fit within token limits."""
    while count_tokens("\n".join([f"User: {msg['user']}\nAssistant: {msg['assistant']}" for msg in conversation_history])) > max_tokens:
        conversation_history.pop(0)  # Remove the oldest interaction
    return conversation_history

@app.route("/")
def serve_index():
    return send_from_directory(directory=".", path = "index.html")

@app.route("/upload", methods = ["POST"])
def upload_document():
    '''Endpoint to upload a PDF document'''
    global vector_store

    # Get the uploaded file
    file = request.files['file']
    if not file:
        return jasonify({"error": "No file uploaded"}), 400

    
    # Save the file temporarily
    #with NamedTemporaryFile(delete = False, suffix = ".pdf") as temp_file:
    #    file.save(temp_file.name)
    #    temp_file_path = temp_file.name
    #    print(f"Temporary file saved at: {temp_file_path}")

    
    try:
        #temp_file_path = f"/tmp/{file.filename}"
        #file.save(temp_file_path)
        #print (f"Temporary file saved at: {temp_file_path}")  # Debugging line

        with pdfplumber.open(file) as pdf:
            docs = [
                Document(page_content=page.extract_text(), metadata={"page": i})
                for i, page in enumerate(pdf.pages)
                if page.extract_text()  # Exclude empty pages
            ]

        text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000, chunk_overlap = 200, add_start_index = True
        )
        all_splits = text_splitter.split_documents(docs)


        # Create embeddings and vector store
        embeddings = OpenAIEmbeddings()
        vector_store = InMemoryVectorStore(embeddings)
        vector_store.add_documents(documents = all_splits)

        # Clean up temporary file
        #os.remove(temp_file_path)

        return jsonify({"message": "Document uploaded successfully", "chunks": len(all_splits)})
    
    except Exception as e:
        # Clean up temporary file in case of an error
        #os.remove(temp_file_path)
        return jsonify({"error": str(e)}), 500



@app.route("/ask", methods = ["POST"])
def ask_question():
    '''Endpoint to ask questions about the document'''
    global vector_store, conversation_history

    # Ensure vector store is initialized
    if not vector_store:
        return jsonify({"error": "No document uploaded"}), 400

    # Get the user's question
    data = request.json
    question = data.get("question")

    if not question:
        return jsonify({"error": "No question provided"}), 400

    # Retrieve relevant context
    retrieved_docs = vector_store.similarity_search(question)
    docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

    # Format the conversation history
    formatted_history = "\n".join(
        [f"User: {msg['user']}\nAssistant: {msg['assistant']}" for msg in conversation_history]
        )
    

    PROMPT_TEMPLATE = """
    You are a highly skilled researcher with expertise in machine learning, advanced mathematics, and statistical modeling. Your responses should demonstrate a deep understanding of these domains, providing precise, well-explained answers that include relevant mathematical formulations, citations to foundational concepts, and real-world examples where appropriate. When uncertain, admit the lack of specific knowledge rather than speculating. Aim for a formal tone, as your audience consists of academics and professionals in these fields. Structure responses clearly, starting with an overview before delving into technical details.
    Do not generate your response in markdown format, rather generate it in HTML format such that it is rendered in webpage without issues. But generate math equations/symbols using mathjax format.
    Conversation History: {history}
    Context: {context}
    Question: {question}
    Answer the question based on the above context and history:"""

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    # Create the full prompt
    prompt = prompt_template.format(
        history=formatted_history, context=docs_content, question=question
    )

    # Check token count and truncate if necessary
    total_tokens = count_tokens(prompt)

    if total_tokens > TOKEN_LIMIT:
        conversation_history = truncate_history(conversation_history, TOKEN_LIMIT - count_tokens(docs_content) - 500)
        formatted_history = "\n".join(
            [f"User: {msg['user']}\nAssistant: {msg['assistant']}" for msg in conversation_history]
        )
        prompt = prompt_template.format(
            history=formatted_history, context=docs_content, question=question
        )

    # Generate a response
    try:
        response = llm.invoke(prompt)
        answer = response.content

        # Update conversation history
        conversation_history.append({"user": question, "assistant": answer})

        return jsonify({"answer": answer})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)