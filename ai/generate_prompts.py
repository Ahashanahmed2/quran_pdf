#!/usr/bin/env python3
# ai/generate_prompts.py
"""
Local Prompt Generator
Generates prompts from downloaded Pinecone data (local JSON files)
No external API calls - fully offline
"""

import os
import re
import json
import hashlib
import random
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import logging

# ============ Configuration ============

DATA_DIR = Path(os.environ.get("DATA_DIR", "./data/raw"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "./data/prompts"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============ Prompt Templates ============

SYSTEM_PROMPTS = {
    "tafsir": "আপনি একজন ইসলামিক স্কলার এবং তাফসীর বিশেষজ্ঞ। কুরআনের আয়াতের সঠিক ব্যাখ্যা প্রদান করুন।",
    "hadith": "আপনি একজন মুহাদ্দিস এবং হাদীস বিশেষজ্ঞ। হাদীসের সঠিক ব্যাখ্যা ও মান নির্ণয় করুন।",
    "fiqh": "আপনি একজন ফকীহ এবং ইসলামী আইন বিশেষজ্ঞ। ফিকহী মাসআলার সঠিক সমাধান প্রদান করুন।",
    "aqidah": "আপনি একজন ইসলামী আকীদা বিশেষজ্ঞ। আকীদার সঠিক ব্যাখ্যা প্রদান করুন।",
    "seerah": "আপনি একজন সীরাহ বিশেষজ্ঞ। নবীজির জীবনী সম্পর্কে সঠিক তথ্য প্রদান করুন।",
    "general": "আপনি একজন ইসলামী জ্ঞানের বিশেষজ্ঞ। সঠিক ও নির্ভরযোগ্য তথ্য প্রদান করুন।",
}

QUESTION_TEMPLATES = {
    "what": ["{topic} কী?", "{topic} কাকে বলে?", "{topic} বলতে কী বোঝায়?"],
    "why": ["{topic} কেন?", "{topic} এর কারণ কী?", "{topic} এর গুরুত্ব কী?"],
    "how": ["{topic} কীভাবে?", "{topic} করার পদ্ধতি কী?", "কীভাবে {topic} বুঝবেন?"],
    "explain": ["{topic} ব্যাখ্যা করুন।", "{topic} সম্পর্কে বিস্তারিত বলুন।", "{topic} এর তাফসীর কী?"],
    "compare": ["{topic1} এবং {topic2} এর মধ্যে পার্থক্য কী?"],
    "list": ["{topic} এর বৈশিষ্ট্যসমূহ কী কী?", "{topic} কত প্রকার ও কী কী?"],
    "define": ["{topic} এর সংজ্ঞা দিন।", "{topic} শব্দের অর্থ কী?"],
    "elaborate": ["{topic} সম্পর্কে বিস্তারিত আলোচনা করুন।"],
}

# ============ Data Classes ============

@dataclass
class PromptData:
    id: str
    paradigm: str
    messages: List[Dict] = field(default_factory=list)
    prompt: Optional[str] = None
    chosen: Optional[str] = None
    rejected: Optional[str] = None
    response: Optional[str] = None
    reward: Optional[float] = None
    metadata: Dict = field(default_factory=dict)

# ============ Local Prompt Generator ============

class LocalPromptGenerator:
    """Generate prompts from local JSON files"""
    
    def __init__(self):
        self.chunks = self.load_all_chunks()
        self.stats = defaultdict(int)
        logger.info(f"Loaded {len(self.chunks)} total chunks")
    
    def load_all_chunks(self) -> List[Dict]:
        """Load all chunks from JSONL file or directory"""
        chunks = []
        
        # Try JSONL first
        jsonl_path = DATA_DIR / "pinecone_chunks.jsonl"
        if jsonl_path.exists():
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        chunk = json.loads(line.strip())
                        # Convert scores from 0-100 to 0-1
                        for key in ["quality_score", "completeness_score", "coherence_score", "factual_score"]:
                            if key in chunk and chunk[key] > 1:
                                chunk[key] = chunk[key] / 100
                        chunks.append(chunk)
                    except:
                        pass
            logger.info(f"Loaded {len(chunks)} chunks from {jsonl_path}")
        
        # Also load from namespace JSON files
        for json_file in DATA_DIR.glob("*_chunks.json"):
            with open(json_file, 'r', encoding='utf-8') as f:
                ns_chunks = json.load(f)
                for chunk in ns_chunks:
                    for key in ["quality_score", "completeness_score", "coherence_score", "factual_score"]:
                        if key in chunk and chunk[key] > 1:
                            chunk[key] = chunk[key] / 100
                chunks.extend(ns_chunks)
                logger.info(f"Loaded {len(ns_chunks)} chunks from {json_file}")
        
        # Remove duplicates by ID
        seen = set()
        unique_chunks = []
        for chunk in chunks:
            chunk_id = chunk.get("id", hashlib.md5(chunk.get("text", "").encode()).hexdigest())
            if chunk_id not in seen:
                seen.add(chunk_id)
                chunk["chunk_id"] = chunk_id
                unique_chunks.append(chunk)
        
        return unique_chunks
    
    def filter_chunks(self, quality_threshold: float = 0.6, min_chars: int = 100) -> List[Dict]:
        """Filter chunks by quality"""
        return [
            c for c in self.chunks
            if c.get("quality_score", 0) >= quality_threshold
            and c.get("char_count", 0) >= min_chars
        ]
    
    def filter_chosen_chunks(self) -> List[Dict]:
        """Get chosen chunks"""
        return [c for c in self.chunks if c.get("can_be_chosen", False)]
    
    def filter_rejected_chunks(self) -> List[Dict]:
        """Get rejected chunks"""
        return [
            c for c in self.chunks
            if c.get("can_be_rejected", False) or c.get("quality_score", 0) < 0.4
        ]
    
    def filter_by_namespace(self, namespace: str) -> List[Dict]:
        """Filter by namespace"""
        return [c for c in self.chunks if c.get("namespace") == namespace]
    
    def filter_by_curriculum_stage(self, stage: int) -> List[Dict]:
        """Filter by curriculum stage"""
        return [c for c in self.chunks if c.get("curriculum_stage") == stage]
    
    def extract_topic(self, chunk: Dict) -> str:
        """Extract topic from chunk text"""
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
    
    def generate_question(self, chunk: Dict) -> Tuple[str, str]:
        """Generate question for chunk"""
        topic = self.extract_topic(chunk)
        content_type = chunk.get("content_type", "general")
        
        if content_type in ["ayah", "tafsir"]:
            q_types = ["explain", "what", "elaborate"]
        elif content_type == "hadith":
            q_types = ["explain", "what", "define"]
        elif content_type == "fiqh":
            q_types = ["how", "what", "list"]
        else:
            q_types = ["what", "explain", "define"]
        
        q_type = random.choice(q_types)
        template = random.choice(QUESTION_TEMPLATES[q_type])
        question = template.format(topic=topic)
        
        return question, q_type
    
    def format_answer(self, chunk: Dict) -> str:
        """Format answer from chunk"""
        text = chunk.get("text", "")
        book_name = chunk.get("book_name", "")
        page_number = chunk.get("page_number", 0)
        volume = chunk.get("volume", 1)
        
        if book_name and page_number and "[সূত্র:" not in text:
            text += f"\n\n[সূত্র: {book_name}, খন্ড {volume}, পৃষ্ঠা {page_number}]"
        
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
                    id=hashlib.md5(f"sft_{chunk.get('id', '')}_{question}".encode()).hexdigest()[:16],
                    paradigm="sft",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": answer}
                    ],
                    metadata={
                        "chunk_id": chunk.get("id", ""),
                        "book_name": chunk.get("book_name", ""),
                        "content_type": content_type,
                        "question_type": q_type,
                        "quality_score": chunk.get("quality_score", 0.5)
                    }
                )
                prompts.append(prompt_data)
        
        self.stats["sft"] += len(prompts)
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
                    id=hashlib.md5(f"dpo_{chosen.get('id', '')}_{rejected.get('id', '')}".encode()).hexdigest()[:16],
                    paradigm="dpo",
                    messages=[{"role": "system", "content": system_prompt}],
                    prompt=question,
                    chosen=self.format_answer(chosen),
                    rejected=self.format_answer(rejected),
                    metadata={"content_type": content_type}
                )
                prompts.append(prompt_data)
        
        self.stats["dpo"] += len(prompts)
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
            reward = chunk.get("quality_score", 0.5)
            
            prompt_data = PromptData(
                id=hashlib.md5(f"ppo_{chunk.get('id', '')}_{question}".encode()).hexdigest()[:16],
                paradigm="ppo",
                messages=[{"role": "system", "content": system_prompt}],
                prompt=question,
                response=response,
                reward=round(reward, 3),
                metadata={"content_type": content_type}
            )
            prompts.append(prompt_data)
        
        self.stats["ppo"] += len(prompts)
        return prompts
    
    # ============ Export ============
    
    def export_jsonl(self, prompts: List[PromptData], filename: str):
        """Export prompts to JSONL"""
        filepath = OUTPUT_DIR / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            for p in prompts:
                data = {"id": p.id, "paradigm": p.paradigm, "metadata": p.metadata}
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
                
                json.dump(data, f, ensure_ascii=False)
                f.write('\n')
        
        logger.info(f"Exported {len(prompts)} prompts to {filepath}")
    
    # ============ Main Generator ============
    
    def generate_all(self):
        """Generate all prompts"""
        logger.info("=" * 70)
        logger.info("Starting Local Prompt Generation")
        logger.info(f"Total chunks available: {len(self.chunks)}")
        logger.info("=" * 70)
        
        # Filter data
        high_quality = self.filter_chunks(quality_threshold=0.6)[:500]
        chosen = self.filter_chosen_chunks()[:200]
        rejected = self.filter_rejected_chunks()[:200]
        
        logger.info(f"High quality: {len(high_quality)}")
        logger.info(f"Chosen: {len(chosen)}")
        logger.info(f"Rejected: {len(rejected)}")
        
        all_prompts = {}
        
        # SFT
        logger.info("\n[1/6] Generating SFT prompts...")
        all_prompts["sft"] = self.generate_sft_prompts(high_quality)
        
        # DPO & variants
        if chosen and rejected:
            logger.info("\n[2/6] Generating DPO/ORPO/SimPO/CPO prompts...")
            dpo_prompts = self.generate_dpo_prompts(chosen, rejected)
            all_prompts["dpo"] = dpo_prompts
            all_prompts["orpo"] = self.generate_dpo_prompts(chosen[:100], rejected[:100])
            all_prompts["simpo"] = self.generate_dpo_prompts(chosen[:100], rejected[:100])
            all_prompts["cpo"] = self.generate_dpo_prompts(chosen[:100], rejected[:100])
        
        # PPO
        logger.info("\n[3/6] Generating PPO prompts...")
        all_prompts["ppo"] = self.generate_ppo_prompts(high_quality[:200])
        
        # Curriculum
        logger.info("\n[4/6] Generating Curriculum prompts...")
        for stage in range(1, 6):
            stage_chunks = self.filter_by_curriculum_stage(stage)[:100]
            if stage_chunks:
                all_prompts[f"curriculum_stage_{stage}"] = self.generate_sft_prompts(stage_chunks, num_per_chunk=2)
        
        # Category-specific
        logger.info("\n[5/6] Generating category-specific prompts...")
        for ns in ["tafsir", "hadith", "fiqh"]:
            ns_chunks = self.filter_by_namespace(ns)[:100]
            if ns_chunks:
                all_prompts[f"{ns}_sft"] = self.generate_sft_prompts(ns_chunks, num_per_chunk=2)
        
        # Export
        logger.info("\n[6/6] Exporting all prompts...")
        for name, prompts in all_prompts.items():
            if prompts:
                self.export_jsonl(prompts, f"{name}_prompts.jsonl")
        
        # Save stats
        stats_file = OUTPUT_DIR / "generation_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(dict(self.stats), f, indent=2, ensure_ascii=False)
        
        logger.info("\n" + "=" * 70)
        logger.info("✅ Prompt Generation Complete!")
        logger.info("=" * 70)
        for name, count in sorted(self.stats.items()):
            logger.info(f"  {name:20s}: {count:6d} prompts")
        logger.info(f"  {'TOTAL':20s}: {sum(self.stats.values()):6d} prompts")


def main():
    generator = LocalPromptGenerator()
    generator.generate_all()

if __name__ == "__main__":
    main()