#!/usr/bin/env python3
# generate_prompts.py
"""
Prompt/Data Generator ONLY - No Training
Generates training data from MongoDB chunks
Output: JSONL files for each training paradigm
"""

import os
import re
import json
import hashlib
import random
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import logging

from pymongo import MongoClient

# ============ Configuration ============

MONGODB_URI = os.environ.get("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB = os.environ.get("MONGODB_DB", "islamic_library")
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "./data"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============ Enums ============

class TrainingParadigm(str, Enum):
    SFT = "sft"
    DPO = "dpo"
    PPO = "ppo"
    RLHF = "rlhf"
    KTO = "kto"
    ORPO = "orpo"
    SIMPO = "simpo"
    CPO = "cpo"
    AGENTIC = "agentic"
    CURRICULUM = "curriculum"

# ============ Prompt Templates ============

SYSTEM_PROMPTS = {
    "tafsir": "আপনি একজন ইসলামিক স্কলার এবং তাফসীর বিশেষজ্ঞ। কুরআনের আয়াতের সঠিক ব্যাখ্যা প্রদান করুন।",
    "hadith": "আপনি একজন মুহাদ্দিস এবং হাদীস বিশেষজ্ঞ। হাদীসের সঠিক ব্যাখ্যা ও মান নির্ণয় করুন।",
    "fiqh": "আপনি একজন ফকীহ এবং ইসলামী আইন বিশেষজ্ঞ। ফিকহী মাসআলার সঠিক সমাধান প্রদান করুন।",
    "aqidah": "আপনি একজন ইসলামী আকীদা বিশেষজ্ঞ। আকীদার সঠিক ব্যাখ্যা প্রদান করুন।",
    "general": "আপনি একজন ইসলামী জ্ঞানের বিশেষজ্ঞ। সঠিক ও নির্ভরযোগ্য তথ্য প্রদান করুন।",
}

QUESTION_TEMPLATES = {
    "what": ["{topic} কী?", "{topic} কাকে বলে?", "{topic} বলতে কী বোঝায়?"],
    "why": ["{topic} কেন?", "{topic} এর কারণ কী?", "{topic} এর গুরুত্ব কী?"],
    "how": ["{topic} কীভাবে?", "{topic} করার পদ্ধতি কী?", "কীভাবে {topic} বুঝবেন?"],
    "explain": ["{topic} ব্যাখ্যা করুন।", "{topic} সম্পর্কে বিস্তারিত বলুন।", "{topic} এর তাফসীর কী?"],
    "compare": ["{topic1} এবং {topic2} এর মধ্যে পার্থক্য কী?", "{topic1} ও {topic2} এর তুলনা করুন।"],
    "list": ["{topic} এর বৈশিষ্ট্যসমূহ কী কী?", "{topic} কত প্রকার ও কী কী?"],
    "define": ["{topic} এর সংজ্ঞা দিন।", "{topic} শব্দের অর্থ কী?"],
    "elaborate": ["{topic} সম্পর্কে বিস্তারিত আলোচনা করুন।", "{topic} নিয়ে বিস্তারিত বলুন।"],
}

# ============ Data Classes ============

@dataclass
class PromptData:
    """Pure prompt data - no training logic"""
    id: str
    paradigm: str
    messages: List[Dict] = field(default_factory=list)
    prompt: Optional[str] = None
    chosen: Optional[str] = None
    rejected: Optional[str] = None
    response: Optional[str] = None
    reward: Optional[float] = None
    rating: Optional[float] = None
    metadata: Dict = field(default_factory=dict)

# ============ Prompt Generator ============

class PromptGenerator:
    """Generate prompts ONLY - No training"""
    
    def __init__(self):
        self.client = MongoClient(MONGODB_URI)
        self.db = self.client[MONGODB_DB]
        self.stats = defaultdict(int)
    
    def get_chunks(self, quality_threshold: float = 0.6, limit: int = 1000) -> List[Dict]:
        """Get chunks from MongoDB"""
        chunks = list(self.db["chunks"].find({
            "quality_score": {"$gte": quality_threshold},
            "char_count": {"$gte": 100, "$lte": 1000}
        }).limit(limit))
        logger.info(f"Retrieved {len(chunks)} chunks")
        return chunks
    
    def get_negative_chunks(self, limit: int = 500) -> List[Dict]:
        """Get low quality chunks for negative samples"""
        chunks = list(self.db["chunks"].find({
            "$or": [
                {"quality_score": {"$lt": 0.4}},
                {"can_be_rejected": True}
            ]
        }).limit(limit))
        return chunks
    
    def extract_topic(self, chunk: Dict) -> str:
        """Extract topic from chunk"""
        text = chunk.get("text", "")
        
        patterns = [
            r'(?:সূরা|সুরা|سورة)\s*([^\s।]+)',
            r'(?:আয়াত|آیت|آية)\s*([^\s।]+)',
            r'([^\s।]+)\s*(?:এর|সম্পর্কে|বিষয়ে)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()
        
        first_sentence = text.split('।')[0][:50]
        return first_sentence if first_sentence else "ইসলামী জ্ঞান"
    
    def extract_topics(self, chunk: Dict) -> List[str]:
        """Extract multiple topics"""
        topics = []
        text = chunk.get("text", "")
        
        if chunk.get("contains_ayah"):
            match = re.search(r'(?:সূরা|سورة)\s*([^\s।]+)', text)
            if match:
                topics.append(f"সূরা {match.group(1)}")
        
        if chunk.get("contains_hadith"):
            topics.append("হাদীস")
        
        keywords = chunk.get("keywords", [])
        if keywords:
            topics.extend(keywords[:2])
        
        return topics if topics else ["ইসলামী জ্ঞান"]
    
    def generate_question(self, chunk: Dict) -> Tuple[str, str]:
        """Generate a single question"""
        topic = self.extract_topic(chunk)
        content_type = chunk.get("content_type", "general")
        
        if content_type in ["ayah", "tafsir"]:
            q_types = ["explain", "what", "elaborate"]
        elif content_type in ["hadith"]:
            q_types = ["explain", "what", "define"]
        elif content_type in ["fiqh"]:
            q_types = ["how", "what", "list"]
        else:
            q_types = ["what", "explain", "define"]
        
        q_type = random.choice(q_types)
        template = random.choice(QUESTION_TEMPLATES[q_type])
        
        topics = self.extract_topics(chunk)
        if q_type == "compare" and len(topics) >= 2:
            question = template.format(topic1=topics[0], topic2=topics[1])
        else:
            question = template.format(topic=topic)
        
        return question, q_type
    
    def format_answer(self, chunk: Dict) -> str:
        """Format answer from chunk"""
        text = chunk.get("text", "")
        book_name = chunk.get("book_name", "")
        page_number = chunk.get("page_number", "")
        
        if book_name and page_number and "[সূত্র:" not in text:
            text += f"\n\n[সূত্র: {book_name}, পৃষ্ঠা {page_number}]"
        
        return text
    
    # ============ SFT Prompts ============
    
    def generate_sft_prompts(self, chunks: List[Dict], num_per_chunk: int = 3) -> List[PromptData]:
        """Generate SFT prompts"""
        prompts = []
        
        for chunk in chunks:
            content_type = chunk.get("content_type", "general")
            system_prompt = SYSTEM_PROMPTS.get(content_type, SYSTEM_PROMPTS["general"])
            
            for _ in range(num_per_chunk):
                question, q_type = self.generate_question(chunk)
                answer = self.format_answer(chunk)
                
                prompt_data = PromptData(
                    id=hashlib.md5(f"sft_{chunk['chunk_id']}_{question}".encode()).hexdigest()[:16],
                    paradigm="sft",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": answer}
                    ],
                    metadata={
                        "chunk_id": chunk["chunk_id"],
                        "book_name": chunk.get("book_name", ""),
                        "page_number": chunk.get("page_number", 0),
                        "content_type": content_type,
                        "question_type": q_type,
                        "quality_score": chunk.get("quality_score", 0.5)
                    }
                )
                prompts.append(prompt_data)
        
        self.stats["sft"] = len(prompts)
        return prompts
    
    # ============ DPO Prompts ============
    
    def generate_dpo_prompts(self, chosen_chunks: List[Dict], rejected_chunks: List[Dict]) -> List[PromptData]:
        """Generate DPO prompts"""
        prompts = []
        
        chosen_by_type = defaultdict(list)
        for c in chosen_chunks:
            chosen_by_type[c.get("content_type", "general")].append(c)
        
        rejected_by_type = defaultdict(list)
        for r in rejected_chunks:
            rejected_by_type[r.get("content_type", "general")].append(r)
        
        for content_type in chosen_by_type:
            chosen_list = chosen_by_type[content_type]
            rejected_list = rejected_by_type.get(content_type, [])
            
            for i, chosen in enumerate(chosen_list):
                if i >= len(rejected_list):
                    break
                
                rejected = rejected_list[i]
                
                topic = self.extract_topic(chosen)
                question = f"{topic} ব্যাখ্যা করুন।"
                
                system_prompt = SYSTEM_PROMPTS.get(content_type, SYSTEM_PROMPTS["general"])
                
                prompt_data = PromptData(
                    id=hashlib.md5(f"dpo_{chosen['chunk_id']}_{rejected['chunk_id']}".encode()).hexdigest()[:16],
                    paradigm="dpo",
                    messages=[{"role": "system", "content": system_prompt}],
                    prompt=question,
                    chosen=self.format_answer(chosen),
                    rejected=self.format_answer(rejected),
                    metadata={
                        "chosen_chunk": chosen["chunk_id"],
                        "rejected_chunk": rejected["chunk_id"],
                        "content_type": content_type
                    }
                )
                prompts.append(prompt_data)
        
        self.stats["dpo"] = len(prompts)
        return prompts
    
    # ============ PPO Prompts ============
    
    def generate_ppo_prompts(self, chunks: List[Dict]) -> List[PromptData]:
        """Generate PPO prompts"""
        prompts = []
        
        for chunk in chunks:
            content_type = chunk.get("content_type", "general")
            system_prompt = SYSTEM_PROMPTS.get(content_type, SYSTEM_PROMPTS["general"])
            
            question, _ = self.generate_question(chunk)
            response = self.format_answer(chunk)
            
            reward = (
                chunk.get("quality_score", 0.5) * 0.4 +
                chunk.get("completeness_score", 0.5) * 0.3 +
                chunk.get("factual_score", 0.5) * 0.3
            )
            
            prompt_data = PromptData(
                id=hashlib.md5(f"ppo_{chunk['chunk_id']}_{question}".encode()).hexdigest()[:16],
                paradigm="ppo",
                messages=[{"role": "system", "content": system_prompt}],
                prompt=question,
                response=response,
                reward=round(reward, 3),
                metadata={
                    "chunk_id": chunk["chunk_id"],
                    "content_type": content_type
                }
            )
            prompts.append(prompt_data)
        
        self.stats["ppo"] = len(prompts)
        return prompts
    
    # ============ RLHF Prompts ============
    
    def generate_rlhf_prompts(self, chunks: List[Dict]) -> List[PromptData]:
        """Generate RLHF prompts"""
        prompts = []
        
        for chunk in chunks:
            content_type = chunk.get("content_type", "general")
            system_prompt = SYSTEM_PROMPTS.get(content_type, SYSTEM_PROMPTS["general"])
            
            question, _ = self.generate_question(chunk)
            response = self.format_answer(chunk)
            
            quality = chunk.get("quality_score", 0.5)
            rating = round(quality * 4 + 1, 1)
            
            prompt_data = PromptData(
                id=hashlib.md5(f"rlhf_{chunk['chunk_id']}_{question}".encode()).hexdigest()[:16],
                paradigm="rlhf",
                messages=[{"role": "system", "content": system_prompt}],
                prompt=question,
                response=response,
                rating=rating,
                metadata={
                    "chunk_id": chunk["chunk_id"],
                    "content_type": content_type,
                    "simulated_rating": rating
                }
            )
            prompts.append(prompt_data)
        
        self.stats["rlhf"] = len(prompts)
        return prompts
    
    # ============ KTO Prompts ============
    
    def generate_kto_prompts(self, chosen_chunks: List[Dict], rejected_chunks: List[Dict]) -> List[PromptData]:
        """Generate KTO prompts"""
        prompts = []
        
        for chunk in chosen_chunks:
            content_type = chunk.get("content_type", "general")
            system_prompt = SYSTEM_PROMPTS.get(content_type, SYSTEM_PROMPTS["general"])
            
            topic = self.extract_topic(chunk)
            question = f"{topic} সম্পর্কে বিস্তারিত বলুন।"
            
            prompt_data = PromptData(
                id=hashlib.md5(f"kto_chosen_{chunk['chunk_id']}".encode()).hexdigest()[:16],
                paradigm="kto",
                messages=[{"role": "system", "content": system_prompt}],
                prompt=question,
                chosen=self.format_answer(chunk),
                metadata={"label": True, "chunk_id": chunk["chunk_id"]}
            )
            prompts.append(prompt_data)
        
        for chunk in rejected_chunks[:len(chosen_chunks)]:
            content_type = chunk.get("content_type", "general")
            system_prompt = SYSTEM_PROMPTS.get(content_type, SYSTEM_PROMPTS["general"])
            
            topic = self.extract_topic(chunk)
            question = f"{topic} সম্পর্কে বিস্তারিত বলুন।"
            
            prompt_data = PromptData(
                id=hashlib.md5(f"kto_rejected_{chunk['chunk_id']}".encode()).hexdigest()[:16],
                paradigm="kto",
                messages=[{"role": "system", "content": system_prompt}],
                prompt=question,
                rejected=self.format_answer(chunk),
                metadata={"label": False, "chunk_id": chunk["chunk_id"]}
            )
            prompts.append(prompt_data)
        
        self.stats["kto"] = len(prompts)
        return prompts
    
    # ============ Agentic Prompts ============
    
    def generate_agentic_prompts(self, chunks: List[Dict]) -> List[PromptData]:
        """Generate Agentic prompts"""
        prompts = []
        
        for chunk in chunks:
            content_type = chunk.get("content_type", "general")
            system_prompt = SYSTEM_PROMPTS.get(content_type, SYSTEM_PROMPTS["general"])
            
            topic = self.extract_topic(chunk)
            question = f"{topic} ব্যাখ্যা করুন।"
            
            initial_response = self.format_answer(chunk)
            
            prompt_data = PromptData(
                id=hashlib.md5(f"agentic_{chunk['chunk_id']}".encode()).hexdigest()[:16],
                paradigm="agentic",
                messages=[{"role": "system", "content": system_prompt}],
                prompt=question,
                response=initial_response,
                metadata={
                    "chunk_id": chunk["chunk_id"],
                    "content_type": content_type,
                    "iteration": 1
                }
            )
            prompts.append(prompt_data)
        
        self.stats["agentic"] = len(prompts)
        return prompts
    
    # ============ Export Functions ============
    
    def export_jsonl(self, prompts: List[PromptData], filename: str):
        """Export prompts to JSONL"""
        filepath = OUTPUT_DIR / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            for p in prompts:
                data = {
                    "id": p.id,
                    "paradigm": p.paradigm,
                    "metadata": p.metadata
                }
                
                if p.messages:
                    data["messages"] = p.messages
                if p.prompt:
                    data["prompt"] = p.prompt
                if p.chosen:
                    data["chosen"] = p.chosen
                if p.rejected:
                    data["rejected"] = p.rejected
                if p.response:
                    data["response"] = p.response
                if p.reward is not None:
                    data["reward"] = p.reward
                if p.rating is not None:
                    data["rating"] = p.rating
                
                json.dump(data, f, ensure_ascii=False)
                f.write('\n')
        
        logger.info(f"Exported {len(prompts)} prompts to {filepath}")
    
    def export_all(self, prompts_by_paradigm: Dict[str, List[PromptData]]):
        """Export all prompts"""
        for paradigm, prompts in prompts_by_paradigm.items():
            if prompts:
                self.export_jsonl(prompts, f"{paradigm}_prompts.jsonl")
    
    # ============ Main Generator ============
    
    def generate_all(self):
        """Generate all prompts"""
        logger.info("=" * 70)
        logger.info("Starting Prompt Generation")
        logger.info("=" * 70)
        
        # Fetch data
        logger.info("Fetching chunks from MongoDB...")
        high_quality = self.get_chunks(quality_threshold=0.7, limit=500)
        negative = self.get_negative_chunks(limit=300)
        
        prompts_by_paradigm = {}
        
        # SFT
        logger.info("\n[1/7] Generating SFT prompts...")
        prompts_by_paradigm["sft"] = self.generate_sft_prompts(high_quality)
        
        # DPO
        logger.info("\n[2/7] Generating DPO prompts...")
        prompts_by_paradigm["dpo"] = self.generate_dpo_prompts(high_quality[:200], negative[:200])
        prompts_by_paradigm["orpo"] = self.generate_dpo_prompts(high_quality[:200], negative[:200])
        prompts_by_paradigm["simpo"] = self.generate_dpo_prompts(high_quality[:200], negative[:200])
        prompts_by_paradigm["cpo"] = self.generate_dpo_prompts(high_quality[:200], negative[:200])
        
        # PPO
        logger.info("\n[3/7] Generating PPO prompts...")
        prompts_by_paradigm["ppo"] = self.generate_ppo_prompts(high_quality[:300])
        
        # RLHF
        logger.info("\n[4/7] Generating RLHF prompts...")
        prompts_by_paradigm["rlhf"] = self.generate_rlhf_prompts(high_quality[:300])
        
        # KTO
        logger.info("\n[5/7] Generating KTO prompts...")
        prompts_by_paradigm["kto"] = self.generate_kto_prompts(high_quality[:200], negative[:200])
        
        # Agentic
        logger.info("\n[6/7] Generating Agentic prompts...")
        prompts_by_paradigm["agentic"] = self.generate_agentic_prompts(high_quality[:200])
        
        # Export
        logger.info("\n[7/7] Exporting all prompts...")
        self.export_all(prompts_by_paradigm)
        
        # Save stats
        stats_file = OUTPUT_DIR / "generation_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(dict(self.stats), f, indent=2, ensure_ascii=False)
        
        logger.info("\n" + "=" * 70)
        logger.info("✅ Prompt Generation Complete!")
        logger.info("=" * 70)
        
        for paradigm, count in sorted(self.stats.items()):
            logger.info(f"  {paradigm:15s}: {count:6d} prompts")
        logger.info(f"  {'TOTAL':15s}: {sum(self.stats.values()):6d} prompts")
    
    def close(self):
        self.client.close()


def main():
    generator = PromptGenerator()
    try:
        generator.generate_all()
    finally:
        generator.close()

if __name__ == "__main__":
    main()