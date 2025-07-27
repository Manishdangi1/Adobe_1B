# 🚀 Quick Start Guide - Adobe Challenge 1B

## ⚡ **One-Command Setup**

```bash
# 1. Build the enhanced solution
docker build -t adobe-pdf-analyzer .

# 2. Run all collections (main command)
docker run --rm -it \
  -v "$(pwd)/Collection 1:/app/Collection 1" \
  -v "$(pwd)/Collection 2:/app/Collection 2" \
  -v "$(pwd)/Collection 3:/app/Collection 3" \
  adobe-pdf-analyzer
```

## 🎯 **Essential Commands**

### **Build & Run**
```bash
# Build image
docker build -t adobe-pdf-analyzer .

# Run all collections
docker run --rm -it adobe-pdf-analyzer

# Run single collection
docker run --rm -it -v "$(pwd)/Collection 1:/app/Collection 1" adobe-pdf-analyzer python enhanced_solution.py
```

### **Testing & Verification**
```bash
# Test the setup
docker run --rm -it adobe-pdf-analyzer python test_enhanced_solution.py

# Verify configuration
docker run --rm -it adobe-pdf-analyzer python verify_setup.py

# Check model loading
docker run --rm -it adobe-pdf-analyzer python optimize_models.py
```

### **Development Mode**
```bash
# Install dependencies locally
pip install -r requirements-optimized.txt
python -m spacy download en_core_web_sm

# Run locally
python process_collections.py
python enhanced_solution.py
python test_enhanced_solution.py
```

