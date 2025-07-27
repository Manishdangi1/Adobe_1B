#!/usr/bin/env python3
"""
Performance Monitor for Adobe Challenge 1B

"""

import os
import sys
import time
import psutil
import logging
from pathlib import Path
from contextlib import contextmanager
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """Monitors performance and resource usage"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.memory_usage = []
        self.cpu_usage = []
        self.constraints = {
            'max_processing_time': 60.0,  # seconds
            'max_memory_gb': 4.0,         # GB
            'max_model_size_gb': 1.0      # GB
        }
        
    @contextmanager
    def monitor_execution(self, operation_name: str = "Operation"):
        """Context manager to monitor execution time and resources"""
        self.start_time = time.time()
        start_memory = psutil.virtual_memory().used
        
        logger.info(f"Starting {operation_name}...")
        
        try:
            yield self
        finally:
            self.end_time = time.time()
            end_memory = psutil.virtual_memory().used
            
            processing_time = self.end_time - self.start_time
            memory_used = (end_memory - start_memory) / (1024**3)  # GB
            
            logger.info(f"{operation_name} completed in {processing_time:.2f} seconds")
            logger.info(f"Memory used: {memory_used:.2f} GB")
            
            # Check constraints
            self.check_constraints(processing_time, memory_used)
            
    def check_constraints(self, processing_time: float, memory_used: float):
        """Check if performance meets challenge constraints"""
        logger.info("Checking performance constraints...")
        
        # Check processing time
        if processing_time > self.constraints['max_processing_time']:
            logger.error(f"❌ Processing time ({processing_time:.2f}s) exceeds limit ({self.constraints['max_processing_time']}s)")
        else:
            logger.info(f"✅ Processing time ({processing_time:.2f}s) within limit")
            
        # Check memory usage
        if memory_used > self.constraints['max_memory_gb']:
            logger.error(f"❌ Memory usage ({memory_used:.2f}GB) exceeds limit ({self.constraints['max_memory_gb']}GB)")
        else:
            logger.info(f"✅ Memory usage ({memory_used:.2f}GB) within limit")
            
        # Check model size
        model_size = self.get_model_directory_size()
        if model_size > self.constraints['max_model_size_gb']:
            logger.error(f"❌ Model size ({model_size:.2f}GB) exceeds limit ({self.constraints['max_model_size_gb']}GB)")
        else:
            logger.info(f"✅ Model size ({model_size:.2f}GB) within limit")
            
    def get_model_directory_size(self) -> float:
        """Get size of model directory in GB"""
        models_dir = Path("/app/models")
        if not models_dir.exists():
            return 0.0
            
        total_size = sum(f.stat().st_size for f in models_dir.rglob('*') if f.is_file())
        return total_size / (1024**3)
        
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        return {
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'memory_available_gb': psutil.virtual_memory().available / (1024**3),
            'disk_usage_percent': psutil.disk_usage('/').percent
        }
        
    def log_system_info(self):
        """Log system information"""
        info = self.get_system_info()
        logger.info("System Information:")
        logger.info(f"  CPU cores: {info['cpu_count']}")
        logger.info(f"  Total memory: {info['memory_total_gb']:.2f} GB")
        logger.info(f"  Available memory: {info['memory_available_gb']:.2f} GB")
        logger.info(f"  Disk usage: {info['disk_usage_percent']:.1f}%")
        
    def monitor_resource_usage(self, interval: float = 1.0):
        """Monitor resource usage during execution"""
        while True:
            try:
                memory = psutil.virtual_memory()
                cpu_percent = psutil.cpu_percent(interval=interval)
                
                self.memory_usage.append(memory.used / (1024**3))
                self.cpu_usage.append(cpu_percent)
                
                logger.debug(f"Memory: {memory.used / (1024**3):.2f}GB, CPU: {cpu_percent:.1f}%")
                
            except KeyboardInterrupt:
                break
                
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        if not self.start_time or not self.end_time:
            return {}
            
        processing_time = self.end_time - self.start_time
        
        return {
            'processing_time_seconds': processing_time,
            'memory_peak_gb': max(self.memory_usage) if self.memory_usage else 0,
            'cpu_peak_percent': max(self.cpu_usage) if self.cpu_usage else 0,
            'model_size_gb': self.get_model_directory_size(),
            'constraints_met': {
                'time': processing_time <= self.constraints['max_processing_time'],
                'memory': max(self.memory_usage) <= self.constraints['max_memory_gb'] if self.memory_usage else True,
                'model_size': self.get_model_directory_size() <= self.constraints['max_model_size_gb']
            }
        }

def monitor_function(func):
    """Decorator to monitor function performance"""
    def wrapper(*args, **kwargs):
        monitor = PerformanceMonitor()
        with monitor.monitor_execution(func.__name__):
            return func(*args, **kwargs)
    return wrapper

# Global monitor instance
global_monitor = PerformanceMonitor()

def start_monitoring():
    """Start global performance monitoring"""
    global_monitor.log_system_info()
    logger.info("Performance monitoring started")

def stop_monitoring():
    """Stop global performance monitoring and show summary"""
    summary = global_monitor.get_performance_summary()
    if summary:
        logger.info("Performance Summary:")
        logger.info(f"  Processing time: {summary['processing_time_seconds']:.2f} seconds")
        logger.info(f"  Peak memory: {summary['memory_peak_gb']:.2f} GB")
        logger.info(f"  Peak CPU: {summary['cpu_peak_percent']:.1f}%")
        logger.info(f"  Model size: {summary['model_size_gb']:.2f} GB")
        
        # Check if all constraints are met
        all_met = all(summary['constraints_met'].values())
        if all_met:
            logger.info("✅ All performance constraints met!")
        else:
            logger.error("❌ Some performance constraints violated!")
            
        return all_met
    return False

if __name__ == "__main__":
    # Test the monitor
    monitor = PerformanceMonitor()
    
    with monitor.monitor_execution("Test Operation"):
        time.sleep(2)  # Simulate work
        
    summary = monitor.get_performance_summary()
    print(f"Test completed: {summary}") 