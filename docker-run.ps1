# Docker management script for Adobe PDF Analyzer (PowerShell)

param(
    [Parameter(Position=0)]
    [string]$Command = "help",
    
    [Parameter(Position=1)]
    [string]$Option = ""
)

# Function to print colored output
function Write-Status {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[WARNING] $Message" -ForegroundColor Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

# Function to build the Docker image
function Build-Image {
    Write-Status "Building Docker image..."
    docker build -t adobe-pdf-analyzer .
    Write-Status "Docker image built successfully!"
}

# Function to run the container
function Start-Container {
    param([string]$Mode = "default")
    
    switch ($Mode) {
        "default" {
            Write-Status "Running container in default mode..."
            docker run --rm -it `
                -v "${PWD}/data:/app/data" `
                -v "${PWD}/output:/app/output" `
                -v "${PWD}/logs:/app/logs" `
                -v "${PWD}/Collection 1:/app/Collection 1" `
                adobe-pdf-analyzer
        }
        "interactive" {
            Write-Status "Running container in interactive mode..."
            docker run --rm -it `
                -v "${PWD}/data:/app/data" `
                -v "${PWD}/output:/app/output" `
                -v "${PWD}/logs:/app/logs" `
                -v "${PWD}/Collection 1:/app/Collection 1" `
                adobe-pdf-analyzer /bin/bash
        }
        "api" {
            Write-Status "Running container with FastAPI..."
            docker run --rm -it `
                -p 8000:8000 `
                -v "${PWD}/data:/app/data" `
                -v "${PWD}/output:/app/output" `
                -v "${PWD}/logs:/app/logs" `
                -v "${PWD}/Collection 1:/app/Collection 1" `
                adobe-pdf-analyzer python -m src.main api
        }
        default {
            Write-Error "Unknown mode: $Mode"
            Write-Status "Available modes: default, interactive, api"
            exit 1
        }
    }
}

# Function to run with docker-compose
function Start-Compose {
    param([string]$Profile = "default")
    
    if ($Profile -eq "cache") {
        Write-Status "Running with docker-compose (with Redis cache)..."
        docker-compose --profile cache up --build
    } else {
        Write-Status "Running with docker-compose..."
        docker-compose up --build
    }
}

# Function to clean up
function Remove-DockerResources {
    Write-Status "Cleaning up Docker resources..."
    docker-compose down -v
    docker system prune -f
    Write-Status "Cleanup completed!"
}

# Function to show usage
function Show-Usage {
    Write-Host "Usage: .\docker-run.ps1 [COMMAND] [OPTIONS]" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Commands:" -ForegroundColor Cyan
    Write-Host "  build                    Build the Docker image"
    Write-Host "  run [MODE]              Run the container"
    Write-Host "    MODES:"
    Write-Host "      default              Run in default mode (generate solution)"
    Write-Host "      interactive          Run in interactive mode (bash shell)"
    Write-Host "      api                  Run with FastAPI server"
    Write-Host "  compose [PROFILE]       Run with docker-compose"
    Write-Host "    PROFILES:"
    Write-Host "      default              Run without Redis"
    Write-Host "      cache                Run with Redis cache"
    Write-Host "  cleanup                 Clean up Docker resources"
    Write-Host "  help                    Show this help message"
    Write-Host ""
    Write-Host "Examples:" -ForegroundColor Cyan
    Write-Host "  .\docker-run.ps1 build"
    Write-Host "  .\docker-run.ps1 run default"
    Write-Host "  .\docker-run.ps1 run interactive"
    Write-Host "  .\docker-run.ps1 compose cache"
    Write-Host "  .\docker-run.ps1 cleanup"
}

# Main script logic
switch ($Command) {
    "build" { Build-Image }
    "run" { Start-Container $Option }
    "compose" { Start-Compose $Option }
    "cleanup" { Remove-DockerResources }
    "help" { Show-Usage }
    default { Show-Usage }
} 