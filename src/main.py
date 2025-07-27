"""
Main entry point for Challenge 1b PDF Analysis Application
"""
import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
import click
from loguru import logger

from .core.analysis_engine import AnalysisEngine
from ..config.config import app_config, collection_config

# Configure logging
logging.basicConfig(
    level=getattr(logging, app_config.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class Challenge1bApp:
    """Main application class for Challenge 1b"""
    
    def __init__(self):
        self.engine = AnalysisEngine()
        self.base_dir = app_config.base_dir
    
    async def process_collection(
        self, 
        collection_name: str, 
        challenge_id: str,
        input_file: Path,
        output_file: Path
    ) -> bool:
        """
        Process a single collection
        
        Args:
            collection_name: Name of the collection
            challenge_id: Challenge identifier
            input_file: Path to input JSON file
            output_file: Path to output JSON file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load input configuration
            with open(input_file, 'r', encoding='utf-8') as f:
                input_data = json.load(f)
            
            # Extract configuration
            persona = input_data.get("persona", {}).get("role", "")
            job_to_be_done = input_data.get("job_to_be_done", {}).get("task", "")
            
            if not persona or not job_to_be_done:
                logger.error(f"Missing persona or job_to_be_done in {input_file}")
                return False
            
            # Get collection path
            collection_path = self.base_dir / collection_name
            if not collection_path.exists():
                logger.error(f"Collection path not found: {collection_path}")
                return False
            
            logger.info(f"Processing collection: {collection_name}")
            logger.info(f"Challenge ID: {challenge_id}")
            logger.info(f"Persona: {persona}")
            logger.info(f"Task: {job_to_be_done}")
            
            # Run analysis
            result = await self.engine.analyze_collection(
                collection_path=collection_path,
                challenge_id=challenge_id,
                persona=persona,
                job_to_be_done=job_to_be_done
            )
            
            # Save results
            self.engine.save_analysis_result(result, output_file)
            
            logger.info(f"Analysis completed successfully")
            logger.info(f"Processed {result.total_documents} documents")
            logger.info(f"Extracted {result.total_sections} sections")
            logger.info(f"Processing time: {result.processing_time:.2f} seconds")
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing collection {collection_name}: {e}")
            return False
    
    async def process_all_collections(self) -> Dict[str, bool]:
        """
        Process all collections in the project
        
        Returns:
            Dictionary mapping collection names to success status
        """
        results = {}
        
        # Find all collection directories
        collection_dirs = [d for d in self.base_dir.iterdir() 
                         if d.is_dir() and d.name.startswith("Collection")]
        
        if not collection_dirs:
            logger.warning("No collection directories found")
            return results
        
        logger.info(f"Found {len(collection_dirs)} collection directories")
        
        for collection_dir in collection_dirs:
            collection_name = collection_dir.name
            
            # Look for input file
            input_file = collection_dir / "challenge1b_input.json"
            if not input_file.exists():
                logger.warning(f"No input file found for {collection_name}")
                results[collection_name] = False
                continue
            
            # Determine challenge ID from input file
            try:
                with open(input_file, 'r', encoding='utf-8') as f:
                    input_data = json.load(f)
                challenge_id = input_data.get("challenge_info", {}).get("challenge_id", "")
            except Exception as e:
                logger.error(f"Error reading input file for {collection_name}: {e}")
                results[collection_name] = False
                continue
            
            if not challenge_id:
                logger.warning(f"No challenge ID found in {collection_name}")
                results[collection_name] = False
                continue
            
            # Set output file path
            output_file = collection_dir / "challenge1b_output.json"
            
            # Process collection
            success = await self.process_collection(
                collection_name=collection_name,
                challenge_id=challenge_id,
                input_file=input_file,
                output_file=output_file
            )
            
            results[collection_name] = success
        
        return results
    
    async def cleanup(self):
        """Cleanup resources"""
        await self.engine.cleanup()

@click.group()
def cli():
    """Challenge 1b PDF Analysis Application"""
    pass

@cli.command()
@click.option('--collection', '-c', help='Specific collection to process')
@click.option('--all', 'process_all', is_flag=True, help='Process all collections')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def process(collection: Optional[str], process_all: bool, verbose: bool):
    """Process PDF collections"""
    if verbose:
        logger.add(lambda msg: print(msg, end=""), level="DEBUG")
    
    async def run():
        app = Challenge1bApp()
        
        try:
            if process_all:
                logger.info("Processing all collections...")
                results = await app.process_all_collections()
                
                # Print summary
                successful = sum(1 for success in results.values() if success)
                total = len(results)
                logger.info(f"\nProcessing Summary:")
                logger.info(f"Successful: {successful}/{total}")
                
                for collection_name, success in results.items():
                    status = "✓" if success else "✗"
                    logger.info(f"  {status} {collection_name}")
            
            elif collection:
                # Process specific collection
                collection_path = app.base_dir / collection
                if not collection_path.exists():
                    logger.error(f"Collection not found: {collection}")
                    return
                
                input_file = collection_path / "challenge1b_input.json"
                output_file = collection_path / "challenge1b_output.json"
                
                if not input_file.exists():
                    logger.error(f"Input file not found: {input_file}")
                    return
                
                # Determine challenge ID
                with open(input_file, 'r', encoding='utf-8') as f:
                    input_data = json.load(f)
                challenge_id = input_data.get("challenge_info", {}).get("challenge_id", "")
                
                success = await app.process_collection(
                    collection_name=collection,
                    challenge_id=challenge_id,
                    input_file=input_file,
                    output_file=output_file
                )
                
                if success:
                    logger.info(f"✓ Successfully processed {collection}")
                else:
                    logger.error(f"✗ Failed to process {collection}")
            
            else:
                logger.error("Please specify --collection or --all")
                return
        
        finally:
            await app.cleanup()
    
    asyncio.run(run())

@cli.command()
def info():
    """Show application information"""
    logger.info("Challenge 1b PDF Analysis Application")
    logger.info(f"Base directory: {app_config.base_dir}")
    logger.info(f"Collections directory: {app_config.collections_dir}")
    logger.info(f"Output directory: {app_config.output_dir}")
    logger.info(f"Cache directory: {app_config.cache_dir}")
    
    # Show available collections
    collection_dirs = [d for d in app_config.base_dir.iterdir() 
                      if d.is_dir() and d.name.startswith("Collection")]
    
    if collection_dirs:
        logger.info(f"\nAvailable collections:")
        for collection_dir in collection_dirs:
            logger.info(f"  - {collection_dir.name}")
    else:
        logger.info("\nNo collection directories found")

@cli.command()
def setup():
    """Setup the application environment"""
    logger.info("Setting up Challenge 1b application...")
    
    # Create necessary directories
    directories = [
        app_config.output_dir,
        app_config.cache_dir,
        app_config.models_dir
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")
    
    # Create sample collection structure
    sample_collections = [
        "Collection 1",
        "Collection 2", 
        "Collection 3"
    ]
    
    for collection_name in sample_collections:
        collection_dir = app_config.base_dir / collection_name
        collection_dir.mkdir(exist_ok=True)
        
        pdf_dir = collection_dir / "PDFs"
        pdf_dir.mkdir(exist_ok=True)
        
        # Create sample input file
        sample_input = {
            "challenge_info": {
                "challenge_id": f"round_1b_{sample_collections.index(collection_name) + 1:03d}",
                "test_case_name": f"sample_test_{sample_collections.index(collection_name) + 1}"
            },
            "documents": [],
            "persona": {"role": "Sample Persona"},
            "job_to_be_done": {"task": "Sample task description"}
        }
        
        input_file = collection_dir / "challenge1b_input.json"
        if not input_file.exists():
            with open(input_file, 'w', encoding='utf-8') as f:
                json.dump(sample_input, f, indent=2)
            logger.info(f"Created sample input file: {input_file}")
    
    logger.info("Setup completed successfully!")

if __name__ == "__main__":
    cli() 