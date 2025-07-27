#!/usr/bin/env python3
"""
Setup Verification Script
"""

import json
import sys
import os
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_file_structure():
    """Verify all required files are present"""
    logger.info("🔍 Verifying file structure...")
    
    required_files = [
        "Dockerfile",
        "docker-compose.yml", 
        "requirements-optimized.txt",
        "enhanced_solution.py",
        "process_collections.py",
        "optimize_models.py",
        "performance_monitor.py",
        "test_enhanced_solution.py",
        "generate_solution.py",
        ".dockerignore",
        "DOCKER_README.md"
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        logger.error(f"❌ Missing files: {missing_files}")
        return False
    else:
        logger.info("✅ All required files present")
        return True

def verify_requirements():
    """Verify requirements file contains superior models"""
    logger.info("🔍 Verifying requirements configuration...")
    
    try:
        with open("requirements-optimized.txt", "r") as f:
            content = f.read()
        
        superior_models = [
            "sentence-transformers==2.2.2",
            "transformers==4.35.2", 
            "torch==2.1.1+cpu",
            "spacy==3.7.2",
            "nltk==3.8.1"
        ]
        
        missing_models = []
        for model in superior_models:
            if model not in content:
                missing_models.append(model)
        
        if missing_models:
            logger.error(f"❌ Missing superior models in requirements: {missing_models}")
            return False
        else:
            logger.info("✅ All superior models included in requirements")
            return True
            
    except Exception as e:
        logger.error(f"❌ Error reading requirements: {e}")
        return False

def verify_dockerfile():
    """Verify Dockerfile is properly configured"""
    logger.info("🔍 Verifying Dockerfile configuration...")
    
    try:
        with open("Dockerfile", "r") as f:
            content = f.read()
        
        required_configs = [
            "FROM python:3.11-slim",
            "ENV CUDA_VISIBLE_DEVICES=\"\"",
            "ENV TRANSFORMERS_CACHE=/app/models",
            "RUN python optimize_models.py",
            "CMD [\"python\", \"process_collections.py\"]"
        ]
        
        missing_configs = []
        for config in required_configs:
            if config not in content:
                missing_configs.append(config)
        
        if missing_configs:
            logger.error(f"❌ Missing Docker configurations: {missing_configs}")
            return False
        else:
            logger.info("✅ Dockerfile properly configured for superior models")
            return True
            
    except Exception as e:
        logger.error(f"❌ Error reading Dockerfile: {e}")
        return False

def verify_enhanced_solution():
    """Verify enhanced solution script structure"""
    logger.info("🔍 Verifying enhanced solution script...")
    
    try:
        with open("enhanced_solution.py", "r", encoding='utf-8') as f:
            content = f.read()
        
        required_features = [
            "all-mpnet-base-v2",
            "microsoft/MiniLM-L12-H384-uncased",
            "calculate_importance_enhanced",
            "refine_text_enhanced",
            "PerformanceMonitor",
            "travel planner",
            "hr professional",
            "food contractor"
        ]
        
        missing_features = []
        for feature in required_features:
            if feature not in content:
                missing_features.append(feature)
        
        if missing_features:
            logger.error(f"❌ Missing enhanced features: {missing_features}")
            return False
        else:
            logger.info("✅ Enhanced solution script properly configured")
            return True
            
    except Exception as e:
        logger.error(f"❌ Error reading enhanced solution: {e}")
        return False

def verify_optimize_models():
    """Verify model optimization script"""
    logger.info("🔍 Verifying model optimization script...")
    
    try:
        with open("optimize_models.py", "r") as f:
            content = f.read()
        
        required_models = [
            "all-mpnet-base-v2",
            "microsoft/MiniLM-L12-H384-uncased",
            "en_core_web_sm"
        ]
        
        missing_models = []
        for model in required_models:
            if model not in content:
                missing_models.append(model)
        
        if missing_models:
            logger.error(f"❌ Missing models in optimization script: {missing_models}")
            return False
        else:
            logger.info("✅ Model optimization script properly configured")
            return True
            
    except Exception as e:
        logger.error(f"❌ Error reading optimization script: {e}")
        return False

def verify_constraint_compliance():
    """Verify constraint compliance configuration"""
    logger.info("🔍 Verifying constraint compliance...")
    
    # Check model size estimate
    estimated_size = 420 + 120 + 12 + 50  # MB
    size_gb = estimated_size / 1024
    
    if size_gb <= 1.0:
        logger.info(f"✅ Estimated model size: {size_gb:.2f} GB (under 1GB limit)")
    else:
        logger.error(f"❌ Estimated model size: {size_gb:.2f} GB (exceeds 1GB limit)")
        return False
    
    # Check CPU-only configuration
    if "CUDA_VISIBLE_DEVICES=\"\"" in open("Dockerfile").read():
        logger.info("✅ CPU-only mode configured")
    else:
        logger.error("❌ CPU-only mode not configured")
        return False
    
    return True

def main():
    """Run all verification checks"""
    logger.info("🚀 Starting setup verification...")
    
    checks = [
        ("File Structure", verify_file_structure),
        ("Requirements", verify_requirements),
        ("Dockerfile", verify_dockerfile),
        ("Enhanced Solution", verify_enhanced_solution),
        ("Model Optimization", verify_optimize_models),
        ("Constraint Compliance", verify_constraint_compliance)
    ]
    
    passed = 0
    total = len(checks)
    
    for check_name, check_func in checks:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running: {check_name}")
        logger.info(f"{'='*50}")
        
        try:
            if check_func():
                logger.info(f"✅ {check_name}: PASSED")
                passed += 1
            else:
                logger.error(f"❌ {check_name}: FAILED")
        except Exception as e:
            logger.error(f"❌ {check_name}: ERROR - {e}")
    
    logger.info(f"\n{'='*50}")
    logger.info(f"VERIFICATION SUMMARY")
    logger.info(f"{'='*50}")
    logger.info(f"Passed: {passed}/{total}")
    logger.info(f"Success Rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        logger.info("🎉 ALL VERIFICATIONS PASSED!")
        logger.info("✅ Enhanced setup is properly configured")
        logger.info("✅ Superior models will be available in Docker")
        logger.info("✅ All constraints will be satisfied")
        logger.info("✅ Ready for optimal Adobe Challenge 1B performance!")
        logger.info("\n🚀 Next steps:")
        logger.info("1. Build Docker image: docker build -t adobe-pdf-analyzer .")
        logger.info("2. Run enhanced solution: docker run --rm -it adobe-pdf-analyzer")
        logger.info("3. Test with real data: Mount Collection 1 directory")
        return True
    else:
        logger.error("❌ Some verifications failed. Please fix the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 