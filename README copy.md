# Challenge 1b PDF Analysis Application

A production-ready PDF analysis application that generates solutions in the exact format expected by the Adobe India Hackathon Challenge.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- PyPDF2

### Installation
```bash
pip install -r requirements.txt
```

### Usage
```bash
# Generate solution for all collections
python generate_solution.py

# Or run the main application
python -m src.main
```

## 📁 Project Structure

```
Challenge_1b/
├── generate_solution.py      # Main solution generator
├── requirements.txt          # Python dependencies
├── README.md                # This file
├── Collection 1/            # Adobe Hackathon Challenge
│   ├── challenge1b_input.json
│   ├── challenge1b_output.json
│   └── PDFs/
│       └── 6874faecd848a_Adobe_India_Hackathon_-_Challenge_Doc.pdf
├── src/                     # Core application code
│   ├── main.py             # CLI interface
│   ├── core/               # PDF processing engine
│   ├── models/             # ML models and embeddings
│   ├── utils/              # Utility functions
│   └── api/                # FastAPI web interface
└── config/                 # Configuration files
```

## 🎯 Features

- **PDF Text Extraction**: Multi-engine PDF processing
- **Section Analysis**: Intelligent content segmentation
- **Persona-Based Scoring**: Relevance ranking for specific roles
- **Challenge 1b Format**: Exact JSON output structure
- **Docker Ready**: Containerized deployment
- **Performance Optimized**: Fast processing with caching

## 📊 Output Format

The application generates solutions in the exact Challenge 1b format:

```json
{
  "metadata": {
    "input_documents": ["document.pdf"],
    "persona": "User Role",
    "job_to_be_done": "Task description"
  },
  "extracted_sections": [
    {
      "document": "document.pdf",
      "section_title": "Section Title",
      "importance_rank": 1,
      "page_number": 1
    }
  ],
  "subsection_analysis": [
    {
      "document": "document.pdf",
      "refined_text": "Processed content",
      "page_number": 1
    }
  ]
}
```

## 🔧 Technical Requirements

- **Architecture**: AMD64 (CPU-only)
- **Model Size**: ≤ 200MB
- **Processing Time**: ≤ 60 seconds for document collections
- **Network**: Offline processing (no internet required)
- **Memory**: Optimized for 8 CPUs and 16GB RAM

## 📝 License

This project is part of the Adobe India Hackathon Challenge. 