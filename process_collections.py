#!/usr/bin/env python3
"""
Multi-Collection PDF Analysis Processor
Processes all three collections for Adobe Challenge 1B
"""

import json
import sys
import time
from pathlib import Path
import logging
from enhanced_solution import EnhancedChallenge1bGenerator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CollectionProcessor:
    """Processes multiple collections for Challenge 1B"""
    
    def __init__(self):
        self.generator = EnhancedChallenge1bGenerator()
        
    def process_collection(self, collection_path: Path) -> bool:
        """Process a single collection"""
        try:
            # Find input file
            input_file = collection_path / "challenge1b_input.json"
            if not input_file.exists():
                logger.warning(f"⚠️ Input file not found: {input_file}")
                return False
            
            # Set output file
            output_file = collection_path / "challenge1b_output.json"
            
            # Load input to get collection info
            with open(input_file, 'r', encoding='utf-8') as f:
                input_data = json.load(f)
            
            challenge_info = input_data.get("challenge_info", {})
            persona = input_data.get("persona", {}).get("role", "")
            task = input_data.get("job_to_be_done", {}).get("task", "")
            
            logger.info(f"🔍 Processing Collection: {collection_path.name}")
            logger.info(f"📋 Challenge ID: {challenge_info.get('challenge_id', 'Unknown')}")
            logger.info(f"👥 Persona: {persona}")
            logger.info(f"📝 Task: {task}")
            
            # Process with enhanced solution
            success = self.generator.generate_enhanced_solution(input_file, output_file)
            
            if success:
                logger.info(f"✅ Collection {collection_path.name} processed successfully")
                return True
            else:
                logger.error(f"❌ Failed to process collection {collection_path.name}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error processing collection {collection_path.name}: {e}")
            return False
    
    def process_all_collections(self) -> dict:
        """Process all three collections"""
        logger.info("🚀 Starting Multi-Collection PDF Analysis")
        logger.info("=" * 60)
        
        collections = [
            "Collection 1",  # Travel Planning
            "Collection 2",  # Adobe Acrobat Learning  
            "Collection 3"   # Recipe Collection
        ]
        
        results = {}
        
        for collection_name in collections:
            collection_path = Path(collection_name)
            if collection_path.exists():
                logger.info(f"\n📁 Processing {collection_name}...")
                logger.info("-" * 40)
                
                start_time = time.time()
                success = self.process_collection(collection_path)
                processing_time = time.time() - start_time
                
                results[collection_name] = {
                    "success": success,
                    "processing_time": processing_time,
                    "status": "✅ Completed" if success else "❌ Failed"
                }
                
                logger.info(f"⏱️ Processing time: {processing_time:.2f} seconds")
                logger.info(f"Status: {results[collection_name]['status']}")
            else:
                logger.warning(f"⚠️ Collection directory not found: {collection_name}")
                results[collection_name] = {
                    "success": False,
                    "processing_time": 0,
                    "status": "⚠️ Directory not found"
                }
        
        return results
    
    def generate_summary_report(self, results: dict):
        """Generate a summary report of all collections"""
        logger.info("\n" + "=" * 60)
        logger.info("📊 MULTI-COLLECTION PROCESSING SUMMARY")
        logger.info("=" * 60)
        
        total_collections = len(results)
        successful_collections = sum(1 for r in results.values() if r["success"])
        total_time = sum(r["processing_time"] for r in results.values())
        
        logger.info(f"📁 Total Collections: {total_collections}")
        logger.info(f"✅ Successful: {successful_collections}")
        logger.info(f"❌ Failed: {total_collections - successful_collections}")
        logger.info(f"⏱️ Total Processing Time: {total_time:.2f} seconds")
        logger.info(f"📈 Success Rate: {(successful_collections/total_collections)*100:.1f}%")
        
        logger.info("\n📋 Collection Details:")
        for collection_name, result in results.items():
            logger.info(f"  {collection_name}: {result['status']} ({result['processing_time']:.2f}s)")
        
        # Challenge compliance check
        logger.info("\n🎯 Challenge Compliance:")
        if total_time <= 60:
            logger.info("✅ Total processing time under 60 seconds")
        else:
            logger.warning(f"⚠️ Total processing time exceeds 60 seconds: {total_time:.2f}s")
        
        if successful_collections == total_collections:
            logger.info("✅ All collections processed successfully")
        else:
            logger.warning(f"⚠️ {total_collections - successful_collections} collections failed")
        
        return successful_collections == total_collections

def main():
    """Main function"""
    try:
        processor = CollectionProcessor()
        
        # Process all collections
        results = processor.process_all_collections()
        
        # Generate summary report
        all_successful = processor.generate_summary_report(results)
        
        if all_successful:
            logger.info("\n🎉 ALL COLLECTIONS PROCESSED SUCCESSFULLY!")
            logger.info("🚀 Ready for Adobe Challenge 1B submission!")
            sys.exit(0)
        else:
            logger.error("\n❌ Some collections failed to process")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 