#!/bin/bash

# Docker management script for Adobe PDF Analyzer

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to build the Docker image
build_image() {
    print_status "Building Docker image..."
    docker build -t adobe-pdf-analyzer .
    print_status "Docker image built successfully!"
}

# Function to run the container
run_container() {
    local mode=${1:-default}
    
    case $mode in
        "default")
            print_status "Running container in default mode..."
            docker run --rm -it \
                -v "$(pwd)/data:/app/data" \
                -v "$(pwd)/output:/app/output" \
                -v "$(pwd)/logs:/app/logs" \
                -v "$(pwd)/Collection 1:/app/Collection 1" \
                adobe-pdf-analyzer
            ;;
        "interactive")
            print_status "Running container in interactive mode..."
            docker run --rm -it \
                -v "$(pwd)/data:/app/data" \
                -v "$(pwd)/output:/app/output" \
                -v "$(pwd)/logs:/app/logs" \
                -v "$(pwd)/Collection 1:/app/Collection 1" \
                adobe-pdf-analyzer /bin/bash
            ;;
        "api")
            print_status "Running container with FastAPI..."
            docker run --rm -it \
                -p 8000:8000 \
                -v "$(pwd)/data:/app/data" \
                -v "$(pwd)/output:/app/output" \
                -v "$(pwd)/logs:/app/logs" \
                -v "$(pwd)/Collection 1:/app/Collection 1" \
                adobe-pdf-analyzer python -m src.main api
            ;;
        *)
            print_error "Unknown mode: $mode"
            print_status "Available modes: default, interactive, api"
            exit 1
            ;;
    esac
}

# Function to run with docker-compose
run_compose() {
    local profile=${1:-default}
    
    if [ "$profile" = "cache" ]; then
        print_status "Running with docker-compose (with Redis cache)..."
        docker-compose --profile cache up --build
    else
        print_status "Running with docker-compose..."
        docker-compose up --build
    fi
}

# Function to clean up
cleanup() {
    print_status "Cleaning up Docker resources..."
    docker-compose down -v
    docker system prune -f
    print_status "Cleanup completed!"
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  build                    Build the Docker image"
    echo "  run [MODE]              Run the container"
    echo "    MODES:"
    echo "      default              Run in default mode (generate solution)"
    echo "      interactive          Run in interactive mode (bash shell)"
    echo "      api                  Run with FastAPI server"
    echo "  compose [PROFILE]       Run with docker-compose"
    echo "    PROFILES:"
    echo "      default              Run without Redis"
    echo "      cache                Run with Redis cache"
    echo "  cleanup                 Clean up Docker resources"
    echo "  help                    Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 build"
    echo "  $0 run default"
    echo "  $0 run interactive"
    echo "  $0 compose cache"
    echo "  $0 cleanup"
}

# Main script logic
case ${1:-help} in
    "build")
        build_image
        ;;
    "run")
        run_container "$2"
        ;;
    "compose")
        run_compose "$2"
        ;;
    "cleanup")
        cleanup
        ;;
    "help"|*)
        show_usage
        ;;
esac 