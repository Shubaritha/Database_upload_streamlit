# Document to Neon Database Uploader

This application allows you to upload PDF and Word documents to your Neon Database with vector embeddings for semantic search capabilities.

## Features

- Support for PDF and DOCX formats
- Automatic text extraction
- OpenAI embedding generation
- Vector storage in Neon Database (using pgvector)
- Document chunking for large files
- User-friendly Streamlit interface

## Prerequisites

- Python 3.8 or higher
- OpenAI API key
- Neon Database account and connection string

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd neondatabaseuploading
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the root directory and add your OpenAI API key:
```
OPENAI_API_KEY=your_openai_api_key_here
```

## Running the Application

1. Start the Streamlit app:
```bash
streamlit run app.py
```

2. Open your browser and navigate to the displayed URL (typically http://localhost:8501)

## Usage

1. Upload your PDF or Word document
2. Enter your Neon Database connection string
3. Choose to create a new table or use an existing one
4. Process the document and store it with vector embeddings

## Deployment on Streamlit Cloud

1. Push your code to GitHub
2. Sign in to [Streamlit Cloud](https://streamlit.io/cloud)
3. Create a new app pointing to your repository
4. Add the OPENAI_API_KEY as a secret in the Streamlit Cloud settings
5. Deploy the app

## Notes

- Ensure your Neon Database has pgvector extension enabled
- Large documents will be automatically split into chunks
- Each chunk will have its own vector embedding 
