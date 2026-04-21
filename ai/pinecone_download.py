#!/usr/bin/env python3
# ai/pinecone_download.py
"""
Pinecone Data Downloader
Downloads all chunks from Pinecone to local JSON files
"""

import os
import json
import time
from pathlib import Path
from typing import List, Dict
import logging
from pinecone import Pinecone

# ============ Configuration ============

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "islamic-knowledge")
DOWNLOAD_DIR = Path(os.environ.get("DOWNLOAD_DIR", "./data/raw"))
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============ Pinecone Downloader ============

class PineconeDownloader:
    """Download all data from Pinecone to local storage"""
    
    def __init__(self):
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.index = self.pc.Index(PINECONE_INDEX_NAME)
        self.dummy_vector = [0.0] * 1536
        
    def get_all_namespaces(self) -> List[str]:
        """Get all available namespaces"""
        try:
            stats = self.index.describe_index_stats()
            namespaces = list(stats.get("namespaces", {}).keys())
            logger.info(f"Found {len(namespaces)} namespaces: {namespaces}")
            return namespaces
        except Exception as e:
            logger.error(f"Failed to get namespaces: {e}")
            return ["tafsir", "hadith", "fiqh", "aqidah", "seerah", "general"]
    
    def get_namespace_stats(self, namespace: str) -> Dict:
        """Get vector count for namespace"""
        try:
            stats = self.index.describe_index_stats()
            ns_stats = stats.get("namespaces", {}).get(namespace, {})
            return {
                "vector_count": ns_stats.get("vector_count", 0)
            }
        except:
            return {"vector_count": 0}
    
    def download_namespace(self, namespace: str, max_vectors: int = 10000) -> List[Dict]:
        """Download all vectors from a namespace"""
        chunks = []
        stats = self.get_namespace_stats(namespace)
        total_vectors = stats.get("vector_count", 0)
        
        logger.info(f"Downloading namespace '{namespace}' (estimated {total_vectors} vectors)")
        
        try:
            # Query with dummy vector to get metadata
            results = self.index.query(
                namespace=namespace,
                vector=self.dummy_vector,
                top_k=min(max_vectors, total_vectors),
                include_metadata=True,
                include_values=False  # Don't download embeddings
            )
            
            for match in results.matches:
                metadata = match.metadata
                
                chunk = {
                    "id": match.id,
                    "namespace": namespace,
                    "book_id": metadata.get("book_id", ""),
                    "book_name": metadata.get("book_name", ""),
                    "book_category": metadata.get("book_category", namespace),
                    "volume": metadata.get("volume", 1),
                    "page_number": metadata.get("page_number", 0),
                    "chunk_index": metadata.get("chunk_index", 0),
                    "content_type": metadata.get("content_type", "general"),
                    "text": metadata.get("text_preview", ""),
                    "char_count": metadata.get("char_count", 0),
                    "word_count": metadata.get("word_count", 0),
                    "quality_score": metadata.get("quality_score", 50),
                    "completeness_score": metadata.get("completeness_score", 50),
                    "coherence_score": metadata.get("coherence_score", 50),
                    "factual_score": metadata.get("factual_score", 50),
                    "difficulty_level": metadata.get("difficulty_level", "intermediate"),
                    "difficulty_score": metadata.get("difficulty_score", 50),
                    "ready_for_sft": metadata.get("ready_for_sft", 0) == 1,
                    "ready_for_dpo": metadata.get("ready_for_dpo", 0) == 1,
                    "ready_for_ppo": metadata.get("ready_for_ppo", 0) == 1,
                    "ready_for_rlhf": metadata.get("ready_for_rlhf", 0) == 1,
                    "can_be_chosen": metadata.get("can_be_chosen", 0) == 1,
                    "can_be_rejected": metadata.get("can_be_rejected", 0) == 1,
                    "has_arabic": metadata.get("has_arabic", 0) == 1,
                    "has_bangla": metadata.get("has_bangla", 0) == 1,
                    "has_english": metadata.get("has_english", 0) == 1,
                    "contains_ayah": metadata.get("contains_ayah", 0) == 1,
                    "contains_hadith": metadata.get("contains_hadith", 0) == 1,
                    "contains_reference": metadata.get("contains_reference", 0) == 1,
                    "contains_question": metadata.get("contains_question", 0) == 1,
                    "curriculum_stage": metadata.get("curriculum_stage", 0),
                    "version": metadata.get("version", "1.0"),
                    "processed_at": metadata.get("processed_at", ""),
                    "score": match.score if hasattr(match, 'score') else 0
                }
                chunks.append(chunk)
            
            logger.info(f"Downloaded {len(chunks)} chunks from '{namespace}'")
            
        except Exception as e:
            logger.error(f"Error downloading from '{namespace}': {e}")
        
        return chunks
    
    def download_all_namespaces(self, max_per_namespace: int = 10000) -> Dict[str, List[Dict]]:
        """Download all namespaces"""
        all_data = {}
        namespaces = self.get_all_namespaces()
        
        for ns in namespaces:
            chunks = self.download_namespace(ns, max_per_namespace)
            if chunks:
                all_data[ns] = chunks
            time.sleep(0.5)  # Rate limiting
        
        return all_data
    
    def save_to_json(self, data: Dict[str, List[Dict]], filename: str = "pinecone_export.json"):
        """Save data to JSON file"""
        filepath = DOWNLOAD_DIR / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved to {filepath}")
        
        # Also save as JSONL for easy processing
        jsonl_path = DOWNLOAD_DIR / "pinecone_chunks.jsonl"
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for ns, chunks in data.items():
                for chunk in chunks:
                    json.dump(chunk, f, ensure_ascii=False)
                    f.write('\n')
        
        logger.info(f"Saved JSONL to {jsonl_path}")
    
    def save_namespace_separately(self, data: Dict[str, List[Dict]]):
        """Save each namespace to separate file"""
        for ns, chunks in data.items():
            filepath = DOWNLOAD_DIR / f"{ns}_chunks.json"
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(chunks, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved {ns} to {filepath}")
    
    def generate_summary(self, data: Dict[str, List[Dict]]) -> Dict:
        """Generate download summary"""
        summary = {
            "total_namespaces": len(data),
            "total_chunks": sum(len(chunks) for chunks in data.values()),
            "namespaces": {}
        }
        
        for ns, chunks in data.items():
            summary["namespaces"][ns] = {
                "chunk_count": len(chunks),
                "avg_quality": sum(c.get("quality_score", 0) for c in chunks) / len(chunks) if chunks else 0,
                "sft_ready": sum(1 for c in chunks if c.get("ready_for_sft")),
                "dpo_ready": sum(1 for c in chunks if c.get("ready_for_dpo")),
                "chosen_count": sum(1 for c in chunks if c.get("can_be_chosen")),
                "rejected_count": sum(1 for c in chunks if c.get("can_be_rejected")),
            }
        
        return summary
    
    def run(self):
        """Main download process"""
        logger.info("=" * 70)
        logger.info("Starting Pinecone Data Download")
        logger.info(f"Index: {PINECONE_INDEX_NAME}")
        logger.info(f"Output: {DOWNLOAD_DIR}")
        logger.info("=" * 70)
        
        # Download all data
        data = self.download_all_namespaces(max_per_namespace=10000)
        
        if not data:
            logger.error("No data downloaded!")
            return
        
        # Save data
        self.save_to_json(data)
        self.save_namespace_separately(data)
        
        # Generate summary
        summary = self.generate_summary(data)
        summary_path = DOWNLOAD_DIR / "download_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # Print summary
        logger.info("\n" + "=" * 70)
        logger.info("Download Summary")
        logger.info("=" * 70)
        logger.info(f"Total Namespaces: {summary['total_namespaces']}")
        logger.info(f"Total Chunks: {summary['total_chunks']}")
        logger.info("\nBy Namespace:")
        for ns, stats in summary['namespaces'].items():
            logger.info(f"  {ns:20s}: {stats['chunk_count']:6d} chunks "
                       f"(SFT: {stats['sft_ready']}, DPO: {stats['dpo_ready']})")
        logger.info("=" * 70)


def main():
    downloader = PineconeDownloader()
    downloader.run()

if __name__ == "__main__":
    main()