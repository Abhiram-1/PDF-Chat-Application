# PDF Chat App with AWS Bedrock and Streamlit

This application enables users to interact with PDF documents through a chat interface, leveraging AWS Bedrock for natural language processing and Streamlit for the user interface.

## Features

- PDF document ingestion and processing
- Efficient document retrieval using FAISS vector store
- AWS Bedrock integration for advanced language modeling
- Streamlit-based user-friendly chat interface

## Prerequisites

- AWS account with Bedrock access
- Python 3.7+
- Pip package manager

## Quick Start

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/pdf-chat-app.git
   cd pdf-chat-app
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Configure AWS credentials (via AWS CLI or environment variables)

4. Add PDF files to the `data` folder

5. Run the app:
   ```
   streamlit run app.py
   ```

6. Access the app in your browser (typically at `http://localhost:8501`)

## How It Works

1. PDF documents are loaded from the `data` folder
2. AWS Bedrock creates embeddings for document content
3. Embeddings are stored in a FAISS vector store
4. User queries trigger retrieval of relevant document sections
5. AWS Bedrock Claude model generates responses based on retrieved content
6. Responses are displayed via the Streamlit interface

## Project Structure

- `app.py`: Main application file (Streamlit UI and core logic)
- `requirements.txt`: Python dependencies
- `data/`: PDF storage
- `faiss-index/`: FAISS index storage (auto-generated)
