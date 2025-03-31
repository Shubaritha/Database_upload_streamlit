# Document Processor with Upstage.ai

This application allows you to upload various document types (PDF, DOCX, PPTX, XLSX) and extract their content, including text, tables, and images using Upstage.ai's document processing capabilities.

## Features

- Support for multiple document formats (PDF, DOCX, PPTX, XLSX)
- Text extraction
- Table extraction and display
- Image extraction and preview
- Modern web interface
- Real-time processing feedback

## Prerequisites

- Node.js (v14 or higher)
- npm (v6 or higher)
- Upstage.ai API key

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd document-processor
```

2. Install dependencies:
```bash
npm install
```

3. Create a `.env` file in the root directory and add your Upstage.ai API key:
```
UPSTAGE_API_KEY=your_api_key_here
```

4. Create the required directories:
```bash
mkdir uploads
mkdir public
```

## Running the Application

1. Start the development server:
```bash
npm run dev
```

2. Open your browser and navigate to:
```
http://localhost:3000
```

## Usage

1. Click the "Choose File" button to select a document
2. Click "Process Document" to start the extraction
3. Wait for the processing to complete
4. View the extracted content:
   - Text content will be displayed at the top
   - Tables will be shown in a formatted view
   - Images will be displayed with previews

## Supported File Types

- PDF (.pdf)
- Microsoft Word (.docx)
- Microsoft PowerPoint (.pptx)
- Microsoft Excel (.xlsx)

## Error Handling

The application includes error handling for:
- Invalid file types
- Processing errors
- API connection issues
- File upload failures

## Security

- Files are temporarily stored in the `uploads` directory and automatically deleted after processing
- API keys are stored securely in environment variables
- Input validation is performed on all file uploads

## Contributing

Feel free to submit issues and enhancement requests! 