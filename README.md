# Adobe Challenge 1B - Enhanced Multi-Collection PDF Analysis

## 🎯 **Challenge Overview**

**Challenge 1B: Multi-Collection PDF Analysis** - Advanced PDF analysis solution that processes multiple document collections and extracts relevant content based on specific personas and use cases.

**: Recipe Collection (round_1b_001) - Food Contractor persona

## 🚀 **Quick Start**

### **Prerequisites**
- Docker
- Docker Compose (optional)
- At least 4GB RAM available

### **1. Clone and Navigate**
```bash
cd "Adobe 1B"
```

### **2. Build the Enhanced Docker Image**
```bash
# Build with superior models
docker build -t adobe-pdf-analyzer .
```

### **3. Run Multi-Collection Analysis**
```bash
# Process all collections automatically
docker run --rm -it \
  -v "$(pwd)/Collection 1:/app/Collection 1" \
  -v "$(pwd)/Collection 2:/app/Collection 2" \
  -v "$(pwd)/Collection 3:/app/Collection 3" \
  adobe-pdf-analyzer
```

## 📁 **Project Structure**

```
Adobe 1B/
├── 🎯 enhanced_solution.py         # Enhanced solution with superior models
├── 🆕 process_collections.py       # Multi-collection processor
├── 🐳 Dockerfile                   # Optimized Docker configuration
├── ⚡ requirements-optimized.txt   # Superior model dependencies
├── ⚡ optimize_models.py           # Superior model optimization
├── 📊 performance_monitor.py       # Performance monitoring
├── 🔄 generate_solution.py         # Original solution (backup)
├── docker-compose.yml           # Multi-service setup
├── 🚫 .dockerignore                # Build optimization
├── 🐧 docker-run.sh               # Linux/Mac management
├── 🪟 docker-run.ps1              # Windows management
├── 📚 DOCKER_README.md            # Comprehensive documentation
├── 🧪 test_enhanced_solution.py    # Comprehensive testing
├── ✅ verify_setup.py              # Setup verification
├── 📖 README.md                    # This documentation
└── ✅ FINAL_CHECKLIST.md           # Final verification checklist
```


## 🚀 **How to Run**

### **Method 1: Docker (Recommended)**

#### **Build the Image**
```bash
# Build with superior models
docker build -t adobe-pdf-analyzer .
```

#### **Run Multi-Collection Analysis**
```bash
# Process all collections 
docker run --rm -it \
  -v "$(pwd)/Collection 1:/app/Collection 1" \
  -v "$(pwd)/Collection 2:/app/Collection 2" \
  -v "$(pwd)/Collection 3:/app/Collection 3" \
  adobe-pdf-analyzer
```

#### **Run Single Collection**
```bash
# Process specific collection
docker run --rm -it \
  -v "$(pwd)/Collection 1:/app/Collection 1" \
  adobe-pdf-analyzer python enhanced_solution.py
```

### **Method 2: Docker Compose**

#### **Run with Docker Compose**
```bash
# Start all services
docker-compose up --build

# Run with Redis caching (optional)
docker-compose --profile cache up --build
```

### **Method 3: Direct Python (Development)**

#### **Install Dependencies**
```bash
# Install superior models
pip install -r requirements-optimized.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

#### **Run Enhanced Solution**
```bash
# Process all collections
python process_collections.py

# Process single collection
python enhanced_solution.py

# Run tests
python test_enhanced_solution.py

# Verify setup
python verify_setup.py
```

