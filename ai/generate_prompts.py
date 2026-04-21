#!/usr/bin/env python3
# ai/generate_prompts_ultimate.py
"""
Ultimate Advanced Prompt Generator with ALL Training Paradigms
- DPO (chosen vs rejected preference pairs)
- ORPO / SimPO / CPO variants
- PPO (RLHF style with reward modeling)
- Curriculum Learning (Stage 1-5)
- Domain-specific SFT splits
- Original features + Continuous Learning
"""

import os
import re
import json
import hashlib
import random
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Set, Any
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import logging

# ============ Configuration ============

DATA_DIR = Path(os.environ.get("DATA_DIR", "./data/raw"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "./data/prompts"))
FEEDBACK_DIR = Path(os.environ.get("FEEDBACK_DIR", "./data/feedback"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)

# Domain-specific output directories
for domain in ["tafsir", "hadith", "fiqh", "aqidah", "seerah", "history", "general"]:
    (OUTPUT_DIR / domain).mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============ Data Models ============

@dataclass
class BookMetadata:
    """Book information"""
    book_id: str
    book_name: str
    category: str
    author: str
    total_volumes: int
    total_pages: int
    total_chunks: int
    added_at: datetime
    last_updated: datetime
    is_new: bool = True

@dataclass
class UserFeedback:
    """User feedback on responses"""
    feedback_id: str
    query: str
    response: str
    rating: int  # 1-5
    helpful: bool
    accurate: bool
    complete: bool
    user_comment: Optional[str]
    missing_info: List[str]
    follow_up_questions: List[str]
    timestamp: datetime
    session_id: str
    topic: str
    book_references: List[str]

@dataclass
class GapAnalysis:
    """Identified gaps in knowledge/response"""
    gap_id: str
    topic: str
    gap_type: str
    description: str
    severity: float
    frequency: int
    affected_queries: List[str]
    suggested_improvement: str

@dataclass
class PreferencePair:
    """DPO/ORPO/SimPO/CPO preference pair"""
    pair_id: str
    prompt: str
    chosen: str
    rejected: str
    chosen_source: str
    rejected_source: str
    preference_strength: float
    domain: str
    difficulty: str

@dataclass
class PPOTrajectory:
    """PPO training trajectory"""
    trajectory_id: str
    prompt: str
    responses: List[str]
    rewards: List[float]
    final_response: str
    final_reward: float
    domain: str
    steps: int

@dataclass
class CurriculumExample:
    """Curriculum learning example"""
    example_id: str
    stage: int  # 1-5
    topic: str
    question: str
    answer: str
    reasoning_steps: List[str]
    domain: str
    difficulty_score: float

# ============ Multilingual Support ============

MULTILINGUAL_TEMPLATES = {
    "bn": {
        "what": ["{topic} কী?", "{topic} কাকে বলে?"],
        "why": ["{topic} কেন?", "{topic} এর কারণ কী?"],
        "how": ["{topic} কীভাবে?", "{topic} করার পদ্ধতি কী?"],
        "when": ["{topic} কখন?", "{topic} কখন ঘটেছে?"],
        "where": ["{topic} কোথায়?", "{topic} কোথায় অবস্থিত?"],
        "who": ["{topic} কে?", "{topic} কারা?"],
    },
    "ar": {
        "what": ["ما هو {topic}؟", "ما معنى {topic}؟"],
        "why": ["لماذا {topic}؟", "ما سبب {topic}؟"],
        "how": ["كيف {topic}؟", "ما طريقة {topic}؟"],
    },
    "en": {
        "what": ["What is {topic}?", "What does {topic} mean?"],
        "why": ["Why {topic}?", "What is the reason for {topic}?"],
        "how": ["How {topic}?", "How to {topic}?"],
    }
}

# ============ Entity Aliases ============

ENTITY_ALIASES = {
    "আল্লাহ": ["খোদা", "রব", "ইলাহ", "মাবুদ", "الله", "God"],
    "মুহাম্মদ": ["নবী", "রাসুল", "আহমদ", "মুস্তফা", "محمد", "Prophet"],
    "কুরআন": ["ফুরকান", "যিকর", "কিতাব", "قرآن", "Quran"],
    "ইবনে কাসীর": ["ইসমাঈল ইবনে কাসীর", "ইবন কাসীর", "ابن كثير"],
    "বুখারী": ["ইমাম বুখারী", "মুহাম্মদ ইসমাঈল বুখারী", "بخاری"],
    "হানাফী": ["হানাফি", "আবু হানীফা", "حنفي"],
    "শাফেয়ী": ["শাফিঈ", "ইমাম শাফেয়ী", "شافعي"],
    "মালিকী": ["মালিকি", "ইমাম মালিক", "مالكي"],
    "হাম্বলী": ["হাম্বলি", "ইমাম আহমদ", "حنبلي"],
}

# ============ Chain-of-Thought Templates ============

COT_TEMPLATES = {
    "step_by_step": [
        "চলুন ধাপে ধাপে বিশ্লেষণ করি:\n১. প্রথমত, {step1}\n২. দ্বিতীয়ত, {step2}\n৩. তৃতীয়ত, {step3}\nসুতরাং, {conclusion}",
        "এটি বুঝতে কয়েকটি ধাপ অনুসরণ করি:\n• প্রথম ধাপ: {step1}\n• দ্বিতীয় ধাপ: {step2}\n• তৃতীয় ধাপ: {step3}\nউপসংহার: {conclusion}"
    ],
    "reasoning": [
        "যুক্তি: {premise1}\nঅতএব, {premise2}\nসুতরাং, {conclusion}",
        "কারণ ১: {reason1}\nকারণ ২: {reason2}\nসিদ্ধান্ত: {conclusion}"
    ]
}

# ============ Contradiction Detection ============

CONTRADICTION_PATTERNS = {
    "false_claim": "{claim} - এটি সঠিক নয়। সঠিক তথ্য হলো: {correct}",
    "partial_truth": "{claim} আংশিক সত্য, তবে {clarification}",
    "out_of_context": "{claim} - এই বক্তব্যটি প্রসঙ্গ থেকে বিচ্ছিন্ন। পূর্ণাঙ্গ প্রসঙ্গ: {context}"
}

# ============ Temporal Reasoning ============

TEMPORAL_TEMPLATES = {
    "sequence": {
        "before": ["{event} এর আগে কী ঘটেছিল?", "{event} সংঘটিত হওয়ার পূর্বে কী অবস্থা ছিল?"],
        "after": ["{event} এর পরে কী ঘটেছিল?", "{event} সংঘটিত হওয়ার পর কী পরিবর্তন এলো?"],
        "during": ["{event} চলাকালে কী ঘটেছিল?", "{event} চলাকালীন সময়ে কী কী ঘটনা ঘটেছিল?"],
    },
    "timeline": [
        "{topic} এর ঘটনাক্রম বিস্তারিত বলুন।",
        "{topic} এর ইতিহাস ধারাবাহিকভাবে বর্ণনা করুন।"
    ],
    "era": [
        "{topic} কোন যুগের ঘটনা?",
        "{topic} কোন শতাব্দীতে সংঘটিত হয়েছিল?"
    ]
}

# ============ Counterfactual Reasoning ============

COUNTERFACTUAL_TEMPLATES = {
    "what_if": [
        "যদি {condition} হতো, তাহলে কী হতো?",
        "{condition} হলে পরিস্থিতি কেমন হতো?",
        "ধরুন {condition}, তাহলে ফলাফল কী দাঁড়াতো?"
    ],
    "alternative": [
        "{action} না করে যদি {alternative} করা হতো, তাহলে কী হতো?",
        "{action} এর পরিবর্তে অন্য কোনো পন্থা অবলম্বন করলে ফলাফল কী হতো?"
    ]
}

# ============ Multi-Hop Reasoning ============

MULTI_HOP_TEMPLATES = {
    "two_hop": [
        "{topic1} এর সাথে {topic2} এর সম্পর্ক কী?",
        "{topic1} কিভাবে {topic2} এর সাথে সংযুক্ত?",
        "{topic1} এবং {topic2} এর মধ্যে যোগসূত্র কী?"
    ],
    "three_hop": [
        "{topic1}, {topic2}, এবং {topic3} এর মধ্যে পারস্পরিক সম্পর্ক কী?",
        "{topic1} → {topic2} → {topic3} এই ধারায় সংযোগ ব্যাখ্যা করুন।"
    ],
    "bridging": [
        "{topic1} থেকে {topic2} তে পৌঁছানোর পথ কী?",
        "{topic1} এবং {topic2} এর মধ্যে সেতুবন্ধন কী?"
    ]
}

# ============ Emotional Intelligence ============

EMOTIONAL_TEMPLATES = {
    "curious": {
        "user": ["জানতে ইচ্ছে করছে, {question}", "একটা প্রশ্ন, {question}"],
        "assistant": ["আপনার কৌতূহল প্রশংসনীয়! {answer}", "জানতে চাওয়া ভালো অভ্যাস। {answer}"]
    },
    "confused": {
        "user": ["বুঝতে পারছি না, {question}", "একটু বিভ্রান্ত হয়েছি, {question}"],
        "assistant": ["চিন্তা করবেন না, আমি পরিষ্কার করে বলছি। {answer}", "বুঝতে অসুবিধা হওয়াটা স্বাভাবিক। {answer}"]
    },
    "skeptical": {
        "user": ["এটা কি সত্যি, {question}?", "বিশ্বাস হচ্ছে না, {question}"],
        "assistant": ["আপনার সন্দেহ যুক্তিযুক্ত। দলিল সহ বলছি: {answer}", "প্রশ্ন করা ভালো। প্রমাণ সহ উত্তর: {answer}"]
    },
    "urgent": {
        "user": ["জরুরি! {question}", "দ্রুত জানতে চাই, {question}"],
        "assistant": ["জরুরি প্রশ্নের জন্য সরাসরি বলছি: {answer}", "তাৎক্ষণিক উত্তর: {answer}"]
    },
    "grateful": {
        "user": ["ধন্যবাদ!", "অনেক উপকার হলো।"],
        "assistant": ["আপনাকে স্বাগতম! আরও কিছু জানতে চাইলে বলুন।", "আপনার উপকারে আসতে পেরে আনন্দিত।"]
    }
}

# ============ Knowledge Graph Traversal ============

GRAPH_TRAVERSAL_TEMPLATES = {
    "parents": [
        "{topic} এর মূল উৎস কী?",
        "{topic} কোথা থেকে এসেছে?",
        "{topic} এর ভিত্তি কী?"
    ],
    "children": [
        "{topic} থেকে কী কী শাখা বের হয়েছে?",
        "{topic} কী কী বিষয়ের জন্ম দিয়েছে?",
        "{topic} এর উপ-বিষয়গুলো কী কী?"
    ],
    "siblings": [
        "{topic} এর সমগোত্রীয় বিষয় কী কী?",
        "{topic} এর মতো আর কী কী বিষয় আছে?",
        "{topic} এর বিকল্প কী কী?"
    ],
    "descendants": [
        "{topic} এর পরবর্তী বিকাশ কী?",
        "{topic} থেকে কী কী বিবর্তিত হয়েছে?"
    ]
}

# ============ Context Window Variations ============

CONTEXT_VARIATIONS = {
    "zero_shot": {
        "instruction": "কোনো অতিরিক্ত তথ্য ছাড়াই শুধু প্রশ্নের উত্তর দিন।",
        "example": "প্রশ্ন: {question}\nউত্তর: {answer}"
    },
    "one_shot": {
        "instruction": "একটি উদাহরণ দেখিয়ে তারপর প্রশ্নের উত্তর দিন।",
        "example": "উদাহরণ:\nপ্রশ্ন: {example_q}\nউত্তর: {example_a}\n\nএবার উত্তর দিন:\nপ্রশ্ন: {question}"
    },
    "few_shot": {
        "instruction": "কয়েকটি উদাহরণ দেখিয়ে তারপর প্রশ্নের উত্তর দিন।",
        "example": "উদাহরণ ১:\nপ্রশ্ন: {q1}\nউত্তর: {a1}\n\nউদাহরণ ২:\nপ্রশ্ন: {q2}\nউত্তর: {a2}\n\nএবার উত্তর দিন:\nপ্রশ্ন: {question}"
    }
}

# ============ Instruction Following ============

INSTRUCTION_TEMPLATES = {
    "format_specific": [
        "উত্তরটি টেবিল আকারে দিন। প্রশ্ন: {question}",
        "উত্তরটি পয়েন্ট আকারে দিন। প্রশ্ন: {question}",
        "উত্তরটি ১০০ শব্দের মধ্যে দিন। প্রশ্ন: {question}",
        "উত্তরটি সহজ ভাষায় দিন। প্রশ্ন: {question}",
        "উত্তরটি পণ্ডিতসুলভ ভাষায় দিন। প্রশ্ন: {question}"
    ],
    "role_specific": [
        "একজন শিক্ষক হিসেবে উত্তর দিন: {question}",
        "একজন বন্ধু হিসেবে উত্তর দিন: {question}",
        "একজন বিশেষজ্ঞ হিসেবে উত্তর দিন: {question}"
    ]
}

# ============ DPO/Preference Templates ============

PREFERENCE_QUESTION_TEMPLATES = [
    "{topic} সম্পর্কে বিস্তারিত বলুন।",
    "{topic} এর ব্যাখ্যা দিন।",
    "{topic} কী? বিস্তারিত উত্তর চাই।",
    "{topic} সম্পর্কে আপনি কী জানেন?",
]

# ============ Curriculum Stage Definitions ============

CURRICULUM_STAGES = {
    1: {"name": "basic_qa", "difficulty": 0.0, "description": "Basic definitions and simple questions"},
    2: {"name": "reasoning", "difficulty": 0.25, "description": "Simple reasoning and explanations"},
    3: {"name": "multi_hop", "difficulty": 0.5, "description": "Multi-step reasoning and connections"},
    4: {"name": "critical", "difficulty": 0.75, "description": "Critical thinking and analysis"},
    5: {"name": "expert", "difficulty": 1.0, "description": "Expert-level comparative analysis"},
}

# ============ New Book Detector ============

class NewBookDetector:
    """Detect and process newly added books"""
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.known_books = self.load_known_books()
    
    def load_known_books(self) -> Dict[str, BookMetadata]:
        registry_file = self.data_dir / "book_registry.json"
        if registry_file.exists():
            with open(registry_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return {
                    bid: BookMetadata(
                        book_id=bid,
                        book_name=b['book_name'],
                        category=b['category'],
                        author=b['author'],
                        total_volumes=b['total_volumes'],
                        total_pages=b['total_pages'],
                        total_chunks=b['total_chunks'],
                        added_at=datetime.fromisoformat(b['added_at']),
                        last_updated=datetime.fromisoformat(b['last_updated']),
                        is_new=False
                    )
                    for bid, b in data.items()
                }
        return {}
    
    def scan_for_new_books(self) -> List[BookMetadata]:
        new_books = []
        
        for json_file in self.data_dir.glob("*_chunks.json"):
            with open(json_file, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
            
            if not chunks:
                continue
            
            books_in_file = {}
            for chunk in chunks:
                book_id = chunk.get("book_id") or hashlib.md5(
                    chunk.get("book_name", "unknown").encode()
                ).hexdigest()[:12]
                
                if book_id not in books_in_file:
                    books_in_file[book_id] = {
                        "book_name": chunk.get("book_name", "Unknown"),
                        "category": chunk.get("book_category", "general"),
                        "author": chunk.get("author", "Unknown"),
                        "volumes": set(),
                        "pages": set(),
                        "chunks": 0
                    }
                
                books_in_file[book_id]["volumes"].add(chunk.get("volume", 1))
                books_in_file[book_id]["pages"].add(
                    f"{chunk.get('volume', 1)}_{chunk.get('page_number', 0)}"
                )
                books_in_file[book_id]["chunks"] += 1
            
            for book_id, info in books_in_file.items():
                if book_id not in self.known_books:
                    book = BookMetadata(
                        book_id=book_id,
                        book_name=info["book_name"],
                        category=info["category"],
                        author=info["author"],
                        total_volumes=len(info["volumes"]),
                        total_pages=len(info["pages"]),
                        total_chunks=info["chunks"],
                        added_at=datetime.now(),
                        last_updated=datetime.now(),
                        is_new=True
                    )
                    new_books.append(book)
                    self.known_books[book_id] = book
        
        self.save_registry()
        return new_books
    
    def save_registry(self):
        registry_file = self.data_dir / "book_registry.json"
        data = {
            bid: {
                "book_name": b.book_name,
                "category": b.category,
                "author": b.author,
                "total_volumes": b.total_volumes,
                "total_pages": b.total_pages,
                "total_chunks": b.total_chunks,
                "added_at": b.added_at.isoformat(),
                "last_updated": b.last_updated.isoformat()
            }
            for bid, b in self.known_books.items()
        }
        with open(registry_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

# ============ Feedback Analyzer ============

class FeedbackAnalyzer:
    """Analyze user feedback to identify improvement areas"""
    
    def __init__(self, feedback_dir: Path):
        self.feedback_dir = feedback_dir
        self.feedbacks = self.load_feedbacks()
    
    def load_feedbacks(self) -> List[UserFeedback]:
        feedbacks = []
        for json_file in self.feedback_dir.glob("*.jsonl"):
            with open(json_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        feedbacks.append(UserFeedback(
                            feedback_id=data.get("id", ""),
                            query=data.get("query", ""),
                            response=data.get("response", ""),
                            rating=data.get("rating", 3),
                            helpful=data.get("helpful", True),
                            accurate=data.get("accurate", True),
                            complete=data.get("complete", True),
                            user_comment=data.get("comment"),
                            missing_info=data.get("missing_info", []),
                            follow_up_questions=data.get("follow_up", []),
                            timestamp=datetime.fromisoformat(data.get("timestamp", datetime.now().isoformat())),
                            session_id=data.get("session_id", ""),
                            topic=data.get("topic", ""),
                            book_references=data.get("references", [])
                        ))
                    except:
                        pass
        return feedbacks
    
    def analyze_gaps(self) -> List[GapAnalysis]:
        gaps = []
        topic_feedback = defaultdict(list)
        for fb in self.feedbacks:
            if fb.topic:
                topic_feedback[fb.topic].append(fb)
        
        for topic, fbs in topic_feedback.items():
            low_ratings = [fb for fb in fbs if fb.rating <= 3]
            if low_ratings:
                gaps.append(GapAnalysis(
                    gap_id=hashlib.md5(f"gap_{topic}_low_rating".encode()).hexdigest()[:12],
                    topic=topic,
                    gap_type="incomplete",
                    description=f"'{topic}' বিষয়ে উত্তর অসম্পূর্ণ বা অসন্তোষজনক",
                    severity=len(low_ratings) / len(fbs) if fbs else 0,
                    frequency=len(low_ratings),
                    affected_queries=[fb.query for fb in low_ratings[:5]],
                    suggested_improvement=f"'{topic}' বিষয়ে আরও বিস্তারিত তথ্য যোগ করুন"
                ))
            
            inaccurate = [fb for fb in fbs if not fb.accurate]
            if inaccurate:
                gaps.append(GapAnalysis(
                    gap_id=hashlib.md5(f"gap_{topic}_inaccurate".encode()).hexdigest()[:12],
                    topic=topic,
                    gap_type="inaccurate",
                    description=f"'{topic}' বিষয়ে ভুল তথ্য প্রদান করা হয়েছে",
                    severity=len(inaccurate) / len(fbs) if fbs else 0,
                    frequency=len(inaccurate),
                    affected_queries=[fb.query for fb in inaccurate[:5]],
                    suggested_improvement=f"'{topic}' এর সঠিক তথ্য যাচাই করে সংশোধন করুন"
                ))
        return gaps
    
    def get_preference_pairs_from_feedback(self) -> List[PreferencePair]:
        """Generate DPO preference pairs from user feedback"""
        pairs = []
        
        # Group feedback by query
        query_feedbacks = defaultdict(list)
        for fb in self.feedbacks:
            query_feedbacks[fb.query].append(fb)
        
        for query, fbs in query_feedbacks.items():
            # Find good vs bad responses
            good_responses = [fb for fb in fbs if fb.rating >= 4]
            bad_responses = [fb for fb in fbs if fb.rating <= 2]
            
            for good in good_responses[:5]:
                for bad in bad_responses[:3]:
                    pair = PreferencePair(
                        pair_id=hashlib.md5(f"dpo_{good.feedback_id}_{bad.feedback_id}".encode()).hexdigest()[:16],
                        prompt=query,
                        chosen=good.response,
                        rejected=bad.response,
                        chosen_source="user_feedback_high_rating",
                        rejected_source="user_feedback_low_rating",
                        preference_strength=(good.rating - bad.rating) / 4.0,
                        domain=good.topic or "general",
                        difficulty="intermediate"
                    )
                    pairs.append(pair)
        
        return pairs
    
    def get_successful_patterns(self) -> List[Dict]:
        patterns = []
        high_ratings = [fb for fb in self.feedbacks if fb.rating >= 4]
        for fb in high_ratings[:50]:
            patterns.append({
                "query": fb.query,
                "response_style": self._analyze_response_style(fb.response),
                "key_elements": self._extract_key_elements(fb.response),
                "topic": fb.topic
            })
        return patterns
    
    def _analyze_response_style(self, response: str) -> str:
        if "উদাহরণ" in response or "যেমন" in response:
            return "with_examples"
        elif "প্রথমত" in response or "দ্বিতীয়ত" in response:
            return "structured"
        elif "সূত্র" in response or "রেফারেন্স" in response or "পৃষ্ঠা" in response:
            return "with_references"
        elif len(response) < 200:
            return "concise"
        elif len(response) > 500:
            return "detailed"
        return "balanced"
    
    def _extract_key_elements(self, response: str) -> List[str]:
        elements = []
        if "সূরা" in response or "আয়াত" in response:
            elements.append("quran_reference")
        if "হাদীস" in response or "বর্ণিত" in response:
            elements.append("hadith_reference")
        if "উদাহরণ" in response:
            elements.append("examples")
        if "মতামত" in response or "মাযহাব" in response:
            elements.append("multiple_opinions")
        return elements

# ============ Ultimate Prompt Generator ============

class UltimatePromptGenerator:
    """Ultimate prompt generator with ALL training paradigms"""
    
    def __init__(self):
        self.chunks = self.load_all_data()
        self.vocabulary = self.build_vocabulary()
        self.word_to_contexts = self.build_word_context_index()
        self.entity_index = self.build_entity_index()
        self.topic_relations = self.build_topic_relations()
        self.timeline_events = self.extract_timeline_events()
        
        self.book_detector = NewBookDetector(DATA_DIR)
        self.feedback_analyzer = FeedbackAnalyzer(FEEDBACK_DIR)
        
        self.stats = defaultdict(int)
        
        logger.info(f"Loaded {len(self.chunks)} chunks, {len(self.vocabulary)} words")
        logger.info(f"Entities: {len(self.entity_index)}, Timeline events: {len(self.timeline_events)}")
    
    def load_all_data(self) -> List[Dict]:
        chunks = []
        jsonl_path = DATA_DIR / "pinecone_chunks.jsonl"
        if jsonl_path.exists():
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        chunks.append(json.loads(line.strip()))
                    except:
                        pass
        return chunks
    
    def build_vocabulary(self) -> Set[str]:
        vocab = set()
        for chunk in self.chunks:
            text = chunk.get("text", "")
            words = re.sub(r'[^\w\u0980-\u09FF\u0600-\u06FF]', ' ', text).split()
            vocab.update([w for w in words if len(w) > 2])
        return vocab
    
    def build_word_context_index(self) -> Dict[str, List[Dict]]:
        index = defaultdict(list)
        for chunk in self.chunks:
            text = chunk.get("text", "")
            words = text.split()
            for i, word in enumerate(words):
                word_clean = re.sub(r'[^\w\u0980-\u09FF\u0600-\u06FF]', '', word)
                if len(word_clean) > 2:
                    start = max(0, i - 10)
                    end = min(len(words), i + 11)
                    index[word_clean].append({
                        "chunk_id": chunk.get("id", ""),
                        "book": chunk.get("book_name", ""),
                        "volume": chunk.get("volume", 1),
                        "page": chunk.get("page_number", 0),
                        "context": ' '.join(words[start:end]),
                        "full_text": text,
                        "category": chunk.get("book_category", "general")
                    })
        return dict(index)
    
    def build_entity_index(self) -> Dict[str, List[str]]:
        index = defaultdict(list)
        for main_entity, aliases in ENTITY_ALIASES.items():
            for alias in aliases:
                index[alias].append(main_entity)
        return dict(index)
    
    def build_topic_relations(self) -> Dict[str, List[Tuple[str, str]]]:
        relations = defaultdict(list)
        relation_types = ["parent_of", "child_of", "related_to", "part_of", "example_of"]
        for topic in list(self.vocabulary)[:100]:
            for rel_type in relation_types:
                relations[topic].append((f"{topic}_related", rel_type))
        return dict(relations)
    
    def extract_timeline_events(self) -> List[Dict]:
        events = []
        timeline_patterns = [
            r'(\d+)\s*(?:হিজরী|হিঃ|AH)',
            r'(?:খ্রিস্টপূর্ব|BCE?)\s*(\d+)',
            r'(\d+)\s*(?:সাল|সনে|খ্রিস্টাব্দ)',
        ]
        for chunk in self.chunks:
            text = chunk.get("text", "")
            for pattern in timeline_patterns:
                matches = re.findall(pattern, text)
                for match in matches:
                    events.append({
                        "year": match,
                        "text": text[:200],
                        "book": chunk.get("book_name", ""),
                        "category": chunk.get("book_category", "general")
                    })
        return events[:100]
    
    def get_word_context(self, word: str) -> Optional[Dict]:
        if word in self.word_to_contexts and self.word_to_contexts[word]:
            return random.choice(self.word_to_contexts[word])
        return None
    
    def get_related_topics(self, topic: str, max_topics: int = 5) -> List[str]:
        return list(self.vocabulary)[:max_topics]
    
    def extract_topic(self, chunk: Dict) -> str:
        text = chunk.get("text", "")
        patterns = [r'(?:সূরা|সুরা)\s*([^\s]+)', r'([^\s]+)\s*(?:নবী|রাসুল)']
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()
        words = text.split()[:3]
        return ' '.join(words) if words else "ইসলামী জ্ঞান"
    
    def find_chunks_by_topic(self, topic: str) -> List[Dict]:
        return [c for c in self.chunks if topic.lower() in c.get("text", "").lower()][:5]
    
    def get_system_prompt(self, category: str) -> str:
        prompts = {
            "tafsir": "আপনি একজন তাফসীর বিশেষজ্ঞ।",
            "hadith": "আপনি একজন হাদীস বিশেষজ্ঞ।",
            "fiqh": "আপনি একজন ফিকহ বিশেষজ্ঞ।",
            "aqidah": "আপনি একজন আকীদা বিশেষজ্ঞ।",
            "seerah": "আপনি একজন সীরাহ বিশেষজ্ঞ।",
            "history": "আপনি একজন ইসলামী ইতিহাসবিদ।",
            "general": "আপনি একজন ইসলামী জ্ঞান বিশেষজ্ঞ।"
        }
        return prompts.get(category, prompts["general"])
    
    def calculate_difficulty_score(self, text: str) -> float:
        word_count = len(text.split())
        arabic_ratio = len(re.findall(r'[\u0600-\u06FF]', text)) / max(len(text), 1)
        score = 0.2
        if word_count > 100:
            score += 0.2
        if word_count > 300:
            score += 0.2
        if arabic_ratio > 0.2:
            score += 0.2
        if re.search(r'(?:তাই|সুতরাং|অতএব|কারণ)', text):
            score += 0.2
        return min(score, 1.0)
    
    def determine_curriculum_stage(self, difficulty_score: float) -> int:
        if difficulty_score < 0.25:
            return 1
        elif difficulty_score < 0.45:
            return 2
        elif difficulty_score < 0.65:
            return 3
        elif difficulty_score < 0.85:
            return 4
        else:
            return 5
    
    # ============ DPO / Preference Pair Generators ============
    
    def generate_dpo_pairs_from_chunks(self, max_pairs: int = 200) -> List[PreferencePair]:
        """Generate DPO preference pairs from chunks"""
        pairs = []
        
        # Group chunks by topic
        topic_chunks = defaultdict(list)
        for chunk in self.chunks:
            topic = self.extract_topic(chunk)
            if topic:
                topic_chunks[topic].append(chunk)
        
        for topic, chunks in topic_chunks.items():
            if len(chunks) < 2:
                continue
            
            # Sort by quality
            chunks.sort(key=lambda x: len(x.get("text", "")), reverse=True)
            
            for i in range(0, len(chunks) - 1, 2):
                if i + 1 >= len(chunks):
                    break
                
                chosen_chunk = chunks[i]
                rejected_chunk = chunks[i + 1]
                
                prompt_template = random.choice(PREFERENCE_QUESTION_TEMPLATES)
                prompt = prompt_template.format(topic=topic)
                
                pair = PreferencePair(
                    pair_id=hashlib.md5(f"dpo_chunk_{chosen_chunk.get('id')}_{rejected_chunk.get('id')}".encode()).hexdigest()[:16],
                    prompt=prompt,
                    chosen=chosen_chunk.get("text", "")[:800],
                    rejected=rejected_chunk.get("text", "")[:800],
                    chosen_source=f"{chosen_chunk.get('book_name', '')} (Quality: High)",
                    rejected_source=f"{rejected_chunk.get('book_name', '')} (Quality: Low)",
                    preference_strength=0.8,
                    domain=chosen_chunk.get("book_category", "general"),
                    difficulty=self.determine_curriculum_stage(
                        self.calculate_difficulty_score(chosen_chunk.get("text", ""))
                    )
                )
                pairs.append(pair)
        
        return pairs[:max_pairs]
    
    def generate_dpo_pairs_from_feedback(self) -> List[PreferencePair]:
        """Generate DPO pairs from user feedback"""
        return self.feedback_analyzer.get_preference_pairs_from_feedback()
    
    def generate_synthetic_preference_pairs(self, max_pairs: int = 100) -> List[PreferencePair]:
        """Generate synthetic preference pairs with variations"""
        pairs = []
        topics = list(self.vocabulary)
        random.shuffle(topics)
        
        for topic in topics[:max_pairs]:
            context = self.get_word_context(topic)
            if not context:
                continue
            
            # Good response (detailed, with references)
            chosen = f"""**{topic} সম্পর্কে বিস্তারিত:**

{context['full_text'][:600]}

**সূত্র:** {context['book']}, খন্ড {context['volume']}, পৃষ্ঠা {context['page']}

**মূল বিষয়:** {topic} ইসলামী জ্ঞানের একটি গুরুত্বপূর্ণ অংশ।"""
            
            # Bad response (too short, no references)
            rejected = f"{topic} একটি গুরুত্বপূর্ণ বিষয়। {context['context']}"
            
            prompt = f"{topic} সম্পর্কে বিস্তারিত বলুন।"
            
            pair = PreferencePair(
                pair_id=hashlib.md5(f"synth_dpo_{topic}".encode()).hexdigest()[:16],
                prompt=prompt,
                chosen=chosen,
                rejected=rejected,
                chosen_source="synthetic_detailed",
                rejected_source="synthetic_brief",
                preference_strength=0.9,
                domain=context.get("category", "general"),
                difficulty="intermediate"
            )
            pairs.append(pair)
        
        return pairs
    
    # ============ PPO / RLHF Generators ============
    
    def generate_ppo_trajectories(self, max_trajectories: int = 100) -> List[PPOTrajectory]:
        """Generate PPO training trajectories with reward modeling"""
        trajectories = []
        topics = list(self.vocabulary)
        random.shuffle(topics)
        
        for topic in topics[:max_trajectories]:
            context = self.get_word_context(topic)
            if not context:
                continue
            
            prompt = f"{topic} সম্পর্কে ব্যাখ্যা করুন।"
            
            # Simulate multiple response attempts with improving quality
            responses = []
            rewards = []
            
            # Attempt 1: Brief response (low reward)
            r1 = f"{topic} একটি ইসলামী পরিভাষা।"
            responses.append(r1)
            rewards.append(0.3)
            
            # Attempt 2: Better response (medium reward)
            r2 = f"{topic} ইসলামী জ্ঞানের একটি গুরুত্বপূর্ণ বিষয়। {context['context']}"
            responses.append(r2)
            rewards.append(0.6)
            
            # Attempt 3: Best response (high reward)
            r3 = f"""**{topic} সম্পর্কে পূর্ণাঙ্গ ব্যাখ্যা:**

{context['full_text'][:500]}

**সূত্র:** {context['book']}, খন্ড {context['volume']}, পৃষ্ঠা {context['page']}

**উপসংহার:** {topic} সম্পর্কে সঠিক জ্ঞান অর্জন করা গুরুত্বপূর্ণ।"""
            responses.append(r3)
            rewards.append(0.9)
            
            trajectory = PPOTrajectory(
                trajectory_id=hashlib.md5(f"ppo_{topic}".encode()).hexdigest()[:16],
                prompt=prompt,
                responses=responses,
                rewards=rewards,
                final_response=r3,
                final_reward=0.9,
                domain=context.get("category", "general"),
                steps=3
            )
            trajectories.append(trajectory)
        
        return trajectories
    
    def generate_reward_model_data(self, max_samples: int = 150) -> List[Dict]:
        """Generate data for reward model training"""
        samples = []
        topics = list(self.vocabulary)
        random.shuffle(topics)
        
        for topic in topics[:max_samples]:
            context = self.get_word_context(topic)
            if not context:
                continue
            
            prompt = f"{topic} কী?"
            
            # Generate responses with different quality levels
            responses = [
                {"text": f"{topic} একটি বিষয়।", "reward": 0.2},
                {"text": f"{topic} গুরুত্বপূর্ণ। {context['context'][:100]}", "reward": 0.5},
                {"text": f"{context['full_text'][:400]}\n\n[সূত্র: {context['book']}]", "reward": 0.9},
            ]
            
            for resp in responses:
                samples.append({
                    "id": hashlib.md5(f"reward_{topic}_{resp['reward']}".encode()).hexdigest()[:16],
                    "prompt": prompt,
                    "response": resp["text"],
                    "reward": resp["reward"],
                    "domain": context.get("category", "general")
                })
        
        return samples
    
    # ============ Curriculum Learning Generators ============
    
    def generate_curriculum_examples(self, max_per_stage: int = 100) -> Dict[int, List[CurriculumExample]]:
        """Generate curriculum learning examples for all 5 stages"""
        curriculum = {1: [], 2: [], 3: [], 4: [], 5: []}
        
        for chunk in self.chunks:
            if sum(len(c) for c in curriculum.values()) >= max_per_stage * 5:
                break
            
            text = chunk.get("text", "")
            difficulty = self.calculate_difficulty_score(text)
            stage = self.determine_curriculum_stage(difficulty)
            
            if len(curriculum[stage]) >= max_per_stage:
                continue
            
            topic = self.extract_topic(chunk)
            domain = chunk.get("book_category", "general")
            
            # Generate stage-appropriate question and answer
            if stage == 1:
                question = f"{topic} কী?"
                answer = f"{topic} হলো {text[:150]}..."
                reasoning = [f"{topic} এর সংজ্ঞা দেওয়া হয়েছে"]
            elif stage == 2:
                question = f"{topic} কেন গুরুত্বপূর্ণ?"
                answer = f"{topic} গুরুত্বপূর্ণ কারণ {text[:250]}..."
                reasoning = [f"{topic} এর গুরুত্ব ব্যাখ্যা করা হয়েছে"]
            elif stage == 3:
                question = f"{topic} এর সাথে অন্য কোন বিষয়ের সম্পর্ক রয়েছে?"
                related = self.get_related_topics(topic, 2)
                answer = f"{topic} এর সাথে {', '.join(related)} এর সম্পর্ক রয়েছে। {text[:350]}..."
                reasoning = [f"{topic} এর সাথে {r} এর সম্পর্ক" for r in related[:2]]
            elif stage == 4:
                question = f"{topic} সম্পর্কে বিভিন্ন মতামত কী?"
                answer = f"এ বিষয়ে বিভিন্ন মতামত রয়েছে। {text[:450]}..."
                reasoning = ["বিভিন্ন দৃষ্টিকোণ বিশ্লেষণ", "তুলনামূলক মূল্যায়ন"]
            else:
                question = f"{topic} এর গভীর বিশ্লেষণ ও তাৎপর্য ব্যাখ্যা করুন।"
                answer = f"{text[:550]}..."
                reasoning = ["গভীর বিশ্লেষণ", "ব্যাপক প্রভাব", "দীর্ঘমেয়াদী তাৎপর্য"]
            
            example = CurriculumExample(
                example_id=hashlib.md5(f"curr_{stage}_{chunk.get('id', topic)}".encode()).hexdigest()[:16],
                stage=stage,
                topic=topic,
                question=question,
                answer=answer,
                reasoning_steps=reasoning,
                domain=domain,
                difficulty_score=difficulty
            )
            curriculum[stage].append(example)
        
        return curriculum
    
    # ============ Original Advanced Prompt Generators ============
    
    def generate_multilingual_prompts(self, max_prompts: int = 200) -> List[Dict]:
        prompts = []
        topics = list(self.vocabulary)
        random.shuffle(topics)
        
        for topic in topics[:max_prompts]:
            for lang, templates in MULTILINGUAL_TEMPLATES.items():
                for q_type, q_templates in templates.items():
                    question = random.choice(q_templates).format(topic=topic)
                    context = self.get_word_context(topic)
                    if not context:
                        continue
                    
                    prompts.append({
                        "id": hashlib.md5(f"multi_{lang}_{topic}_{q_type}".encode()).hexdigest()[:16],
                        "type": "multilingual",
                        "domain": context.get("category", "general"),
                        "messages": [
                            {"role": "system", "content": f"You are a helpful assistant. Respond in {lang}."},
                            {"role": "user", "content": question},
                            {"role": "assistant", "content": context['context']}
                        ],
                        "metadata": {"topic": topic, "language": lang, "question_type": q_type}
                    })
        return prompts
    
    def generate_chain_of_thought_prompts(self, max_prompts: int = 200) -> List[Dict]:
        prompts = []
        topics = list(self.vocabulary)
        random.shuffle(topics)
        
        for topic in topics[:max_prompts]:
            context = self.get_word_context(topic)
            if not context:
                continue
            
            template = random.choice(COT_TEMPLATES["step_by_step"])
            question = f"{topic} সম্পর্কে ব্যাখ্যা করুন။"
            
            answer = template.format(
                step1=f"{topic} এর শাব্দিক অর্থ বোঝা",
                step2=f"{topic} এর প্রাসঙ্গিক ব্যবহার দেখা",
                step3=f"{topic} থেকে শিক্ষা গ্রহণ করা",
                conclusion=context['context']
            )
            
            prompts.append({
                "id": hashlib.md5(f"cot_{topic}".encode()).hexdigest()[:16],
                "type": "chain_of_thought",
                "domain": context.get("category", "general"),
                "messages": [
                    {"role": "system", "content": "ধাপে ধাপে চিন্তা করে উত্তর দিন।"},
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer}
                ],
                "metadata": {"topic": topic, "reasoning_steps": 3}
            })
        return prompts
    
    def generate_contradiction_prompts(self, max_prompts: int = 150) -> List[Dict]:
        prompts = []
        topics = list(self.vocabulary)
        random.shuffle(topics)
        
        for topic in topics[:max_prompts]:
            context = self.get_word_context(topic)
            if not context:
                continue
            
            false_claim = f"{topic} এর কোনো গুরুত্ব নেই।"
            question = f"এই বক্তব্যটি কি সঠিক? '{false_claim}'"
            answer = CONTRADICTION_PATTERNS["false_claim"].format(
                claim=false_claim, correct=context['context']
            )
            
            prompts.append({
                "id": hashlib.md5(f"contra_{topic}".encode()).hexdigest()[:16],
                "type": "contradiction_detection",
                "domain": context.get("category", "general"),
                "messages": [
                    {"role": "system", "content": "ভুল তথ্য শনাক্ত করে সঠিক তথ্য প্রদান করুন।"},
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer}
                ],
                "metadata": {"topic": topic, "false_claim": false_claim}
            })
        return prompts
    
    def generate_temporal_prompts(self, max_prompts: int = 150) -> List[Dict]:
        prompts = []
        for event in self.timeline_events[:max_prompts]:
            for seq_type, templates in TEMPORAL_TEMPLATES["sequence"].items():
                question = random.choice(templates).format(event=event['text'][:50])
                answer = f"{event['text']}\n\nসময়: {event['year']}\nসূত্র: {event['book']}"
                
                prompts.append({
                    "id": hashlib.md5(f"temp_{seq_type}_{event['year']}".encode()).hexdigest()[:16],
                    "type": "temporal_reasoning",
                    "domain": event.get("category", "general"),
                    "messages": [
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": answer}
                    ],
                    "metadata": {"event_year": event['year'], "sequence_type": seq_type}
                })
        return prompts
    
    def generate_counterfactual_prompts(self, max_prompts: int = 100) -> List[Dict]:
        prompts = []
        topics = list(self.vocabulary)
        random.shuffle(topics)
        
        for topic in topics[:max_prompts]:
            context = self.get_word_context(topic)
            if not context:
                continue
            
            template = random.choice(COUNTERFACTUAL_TEMPLATES["what_if"])
            condition = f"{topic} না থাকলে"
            question = template.format(condition=condition)
            answer = f"যদি {condition}, তাহলে ইসলামী জ্ঞানের একটি গুরুত্বপূর্ণ অংশ অনুপস্থিত থাকতো। {context['context']}"
            
            prompts.append({
                "id": hashlib.md5(f"counter_{topic}".encode()).hexdigest()[:16],
                "type": "counterfactual",
                "domain": context.get("category", "general"),
                "messages": [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer}
                ],
                "metadata": {"topic": topic, "condition": condition}
            })
        return prompts
    
    def generate_multihop_prompts(self, max_prompts: int = 100) -> List[Dict]:
        prompts = []
        topics = list(self.vocabulary)
        
        for i in range(0, min(len(topics) - 2, max_prompts)):
            t1, t2, t3 = topics[i], topics[i+1], topics[i+2]
            template = random.choice(MULTI_HOP_TEMPLATES["three_hop"])
            question = template.format(topic1=t1, topic2=t2, topic3=t3)
            
            c1 = self.get_word_context(t1)
            c2 = self.get_word_context(t2)
            c3 = self.get_word_context(t3)
            
            if not all([c1, c2, c3]):
                continue
            
            answer = f"{t1}: {c1['context']}\n\n{t2}: {c2['context']}\n\n{t3}: {c3['context']}\n\nএরা পরস্পর সম্পর্কিত কারণ..."
            
            prompts.append({
                "id": hashlib.md5(f"multihop_{t1}_{t2}_{t3}".encode()).hexdigest()[:16],
                "type": "multi_hop",
                "domain": c1.get("category", "general"),
                "messages": [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer}
                ],
                "metadata": {"topics": [t1, t2, t3], "hops": 3}
            })
        return prompts
    
    def generate_emotional_prompts(self, max_prompts: int = 150) -> List[Dict]:
        prompts = []
        topics = list(self.vocabulary)
        random.shuffle(topics)
        
        for topic in topics[:max_prompts]:
            context = self.get_word_context(topic)
            if not context:
                continue
            
            for emotion, templates in EMOTIONAL_TEMPLATES.items():
                if emotion in ["grateful"]:
                    continue
                
                user_template = random.choice(templates["user"])
                question = user_template.format(question=f"{topic} কী?")
                
                assistant_template = random.choice(templates["assistant"])
                answer = assistant_template.format(answer=context['context'])
                
                prompts.append({
                    "id": hashlib.md5(f"emo_{emotion}_{topic}".encode()).hexdigest()[:16],
                    "type": "emotional_intelligence",
                    "domain": context.get("category", "general"),
                    "messages": [
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": answer}
                    ],
                    "metadata": {"topic": topic, "emotion": emotion}
                })
        return prompts
    
    def generate_graph_traversal_prompts(self, max_prompts: int = 100) -> List[Dict]:
        prompts = []
        topics = list(self.vocabulary)
        random.shuffle(topics)
        
        for topic in topics[:max_prompts]:
            context = self.get_word_context(topic)
            if not context:
                continue
            
            for direction, templates in GRAPH_TRAVERSAL_TEMPLATES.items():
                question = random.choice(templates).format(topic=topic)
                
                if direction == "parents":
                    answer = f"{topic} এর মূল উৎস হলো ইসলামী জ্ঞানের মৌলিক উৎসসমূহ। {context['context']}"
                elif direction == "children":
                    answer = f"{topic} থেকে বিভিন্ন শাখা-প্রশাখা বের হয়েছে। যেমন: {topic} সম্পর্কিত বিভিন্ন মাসআলা।"
                elif direction == "siblings":
                    related = self.get_related_topics(topic, 3)
                    answer = f"{topic} এর সমগোত্রীয় বিষয়: {', '.join(related)}। {context['context']}"
                else:
                    answer = context['context']
                
                prompts.append({
                    "id": hashlib.md5(f"graph_{direction}_{topic}".encode()).hexdigest()[:16],
                    "type": "graph_traversal",
                    "domain": context.get("category", "general"),
                    "messages": [
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": answer}
                    ],
                    "metadata": {"topic": topic, "direction": direction}
                })
        return prompts
    
    def generate_instruction_following_prompts(self, max_prompts: int = 150) -> List[Dict]:
        prompts = []
        topics = list(self.vocabulary)
        random.shuffle(topics)
        
        for topic in topics[:max_prompts]:
            context = self.get_word_context(topic)
            if not context:
                continue
            
            for format_type, templates in INSTRUCTION_TEMPLATES.items():
                question = random.choice(templates).format(question=f"{topic} কী?")
                
                if "টেবিল" in question:
                    answer = f"| বিষয় | তথ্য |\n|-------|------|\n| {topic} | {context['context'][:50]}... |"
                elif "পয়েন্ট" in question:
                    answer = f"• {topic}: {context['context']}"
                elif "১০০ শব্দ" in question:
                    answer = context['context'][:500]
                else:
                    answer = context['context']
                
                prompts.append({
                    "id": hashlib.md5(f"instr_{topic}_{format_type}".encode()).hexdigest()[:16],
                    "type": "instruction_following",
                    "domain": context.get("category", "general"),
                    "messages": [
                        {"role": "system", "content": "নির্দেশনা অনুযায়ী উত্তর দিন।"},
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": answer}
                    ],
                    "metadata": {"topic": topic, "format": format_type}
                })
        return prompts
    
    def generate_context_window_prompts(self, max_prompts: int = 100) -> List[Dict]:
        prompts = []
        topics = list(self.vocabulary)
        random.shuffle(topics)
        
        for topic in topics[:max_prompts]:
            context = self.get_word_context(topic)
            if not context:
                continue
            
            for ctx_type, ctx_config in CONTEXT_VARIATIONS.items():
                if ctx_type == "zero_shot":
                    messages = [
                        {"role": "user", "content": f"{topic} কী?"},
                        {"role": "assistant", "content": context['context'][:200]}
                    ]
                elif ctx_type == "one_shot":
                    messages = [
                        {"role": "user", "content": f"উদাহরণ: ইসলাম কী?\nউত্তর: ইসলাম একটি ধর্ম।"},
                        {"role": "assistant", "content": "বুঝলাম।"},
                        {"role": "user", "content": f"{topic} কী?"},
                        {"role": "assistant", "content": context['context'][:200]}
                    ]
                else:
                    messages = [
                        {"role": "user", "content": f"{topic} কী?"},
                        {"role": "assistant", "content": context['context'][:200]}
                    ]
                
                prompts.append({
                    "id": hashlib.md5(f"ctx_{ctx_type}_{topic}".encode()).hexdigest()[:16],
                    "type": "context_window",
                    "domain": context.get("category", "general"),
                    "messages": messages,
                    "metadata": {"topic": topic, "context_type": ctx_type}
                })
        return prompts
    
    # ============ Continuous Learning Prompt Generators ============
    
    def generate_new_book_prompts(self, books: List[BookMetadata]) -> List[Dict]:
        prompts = []
        for book in books:
            prompts.append({
                "id": hashlib.md5(f"new_book_intro_{book.book_id}".encode()).hexdigest()[:16],
                "type": "new_book_introduction",
                "domain": book.category,
                "messages": [
                    {"role": "system", "content": "আপনি একটি ইসলামী জ্ঞান সহকারী।"},
                    {"role": "user", "content": f"'{book.book_name}' বইটি সম্পর্কে বলুন।"},
                    {"role": "assistant", "content": f"**{book.book_name}** একটি {book.category} বিষয়ক গ্রন্থ।\n\n**লেখক:** {book.author}\n**খন্ড:** {book.total_volumes}\n**পৃষ্ঠা:** {book.total_pages}"}
                ],
                "metadata": {"book_id": book.book_id, "book_name": book.book_name}
            })
        return prompts
    
    def generate_gap_filling_prompts(self, gaps: List[GapAnalysis]) -> List[Dict]:
        prompts = []
        for gap in gaps:
            prompts.append({
                "id": hashlib.md5(f"gap_{gap.gap_id}".encode()).hexdigest()[:16],
                "type": "gap_filling",
                "domain": gap.topic,
                "messages": [
                    {"role": "system", "content": gap.suggested_improvement},
                    {"role": "user", "content": gap.affected_queries[0] if gap.affected_queries else f"{gap.topic} সম্পর্কে বলুন"},
                    {"role": "assistant", "content": f"[উন্নত উত্তর - {gap.description}]"}
                ],
                "metadata": {"gap_type": gap.gap_type, "topic": gap.topic}
            })
        return prompts
    
    def generate_success_pattern_prompts(self, patterns: List[Dict]) -> List[Dict]:
        prompts = []
        for pattern in patterns[:30]:
            prompts.append({
                "id": hashlib.md5(f"success_{pattern.get('topic', '')}".encode()).hexdigest()[:16],
                "type": "success_pattern",
                "domain": pattern.get("topic", "general"),
                "messages": [
                    {"role": "system", "content": f"উত্তর দেওয়ার স্টাইল: {pattern.get('response_style', 'balanced')}"},
                    {"role": "user", "content": pattern["query"]},
                    {"role": "assistant", "content": "[এই স্টাইলে উত্তর]"}
                ],
                "metadata": pattern
            })
        return prompts
    
    def generate_adaptive_finetuning_prompts(self) -> List[Dict]:
        prompts = []
        feedbacks = self.feedback_analyzer.feedbacks
        
        for fb in feedbacks[:30]:
            if fb.rating >= 4:
                prompts.append({
                    "id": hashlib.md5(f"adapt_pos_{fb.feedback_id}".encode()).hexdigest()[:16],
                    "type": "adaptive_positive",
                    "domain": fb.topic or "general",
                    "messages": [
                        {"role": "system", "content": "এই ধরনের উত্তর ভালো রেটিং পেয়েছে।"},
                        {"role": "user", "content": fb.query},
                        {"role": "assistant", "content": fb.response}
                    ],
                    "metadata": {"rating": fb.rating}
                })
        
        return prompts
    
    # ============ Main Execution ============
    
    def generate_all(self):
        """Generate all prompts - ALL paradigms"""
        logger.info("=" * 80)
        logger.info("Starting ULTIMATE Prompt Generation with ALL Training Paradigms")
        logger.info(f"Vocabulary: {len(self.vocabulary)} words")
        logger.info("=" * 80)
        
        all_prompts = {}
        domain_prompts = defaultdict(list)
        
        # Original advanced prompts
        prompt_generators = [
            ("multilingual", self.generate_multilingual_prompts, 200),
            ("chain_of_thought", self.generate_chain_of_thought_prompts, 200),
            ("contradiction", self.generate_contradiction_prompts, 150),
            ("temporal", self.generate_temporal_prompts, 150),
            ("counterfactual", self.generate_counterfactual_prompts, 100),
            ("multihop", self.generate_multihop_prompts, 100),
            ("emotional", self.generate_emotional_prompts, 150),
            ("graph_traversal", self.generate_graph_traversal_prompts, 100),
            ("instruction_following", self.generate_instruction_following_prompts, 150),
            ("context_window", self.generate_context_window_prompts, 100),
        ]
        
        for i, (name, generator, count) in enumerate(prompt_generators, 1):
            logger.info(f"\n[{i}/{len(prompt_generators)}] Generating {name.upper()} prompts...")
            prompts = generator(count)
            all_prompts[name] = prompts
            self.stats[name] = len(prompts)
            
            # Domain-specific categorization
            for p in prompts:
                domain = p.get("domain", "general")
                domain_prompts[domain].append(p)
        
        # ============ NEW: DPO Preference Pairs ============
        logger.info("\n[NEW] Generating DPO Preference Pairs...")
        dpo_pairs = []
        dpo_pairs.extend(self.generate_dpo_pairs_from_chunks(150))
        dpo_pairs.extend(self.generate_dpo_pairs_from_feedback())
        dpo_pairs.extend(self.generate_synthetic_preference_pairs(50))
        self.stats["dpo_pairs"] = len(dpo_pairs)
        
        # ============ NEW: ORPO / SimPO / CPO (DPO variants) ============
        logger.info("\n[NEW] Generating ORPO/SimPO/CPO variants...")
        self.stats["orpo_pairs"] = len(dpo_pairs)
        self.stats["simpo_pairs"] = len(dpo_pairs)
        self.stats["cpo_pairs"] = len(dpo_pairs)
        
        # ============ NEW: PPO Trajectories ============
        logger.info("\n[NEW] Generating PPO Trajectories...")
        ppo_trajectories = self.generate_ppo_trajectories(80)
        self.stats["ppo_trajectories"] = len(ppo_trajectories)
        
        reward_data = self.generate_reward_model_data(100)
        self.stats["reward_model_samples"] = len(reward_data)
        
        # ============ NEW: Curriculum Learning ============
        logger.info("\n[NEW] Generating Curriculum Learning Examples...")
        curriculum = self.generate_curriculum_examples(80)
        for stage, examples in curriculum.items():
            self.stats[f"curriculum_stage_{stage}"] = len(examples)
        
        # ============ Domain-Specific SFT Splits ============
        logger.info("\n[NEW] Creating Domain-Specific SFT Splits...")
        for domain, prompts in domain_prompts.items():
            if domain not in self.stats:
                self.stats[f"domain_{domain}"] = len(prompts)
        
        # Continuous learning prompts
        logger.info("\n--- Continuous Learning Section ---")
        
        new_books = self.book_detector.scan_for_new_books()
        if new_books:
            all_prompts["new_books"] = self.generate_new_book_prompts(new_books)
            self.stats["new_books"] = len(all_prompts["new_books"])
        
        gaps = self.feedback_analyzer.analyze_gaps()
        if gaps:
            all_prompts["gap_filling"] = self.generate_gap_filling_prompts(gaps)
            self.stats["gap_filling"] = len(all_prompts["gap_filling"])
        
        patterns = self.feedback_analyzer.get_successful_patterns()
        if patterns:
            all_prompts["success_patterns"] = self.generate_success_pattern_prompts(patterns)
            self.stats["success_patterns"] = len(all_prompts["success_patterns"])
        
        adaptive = self.generate_adaptive_finetuning_prompts()
        if adaptive:
            all_prompts["adaptive_finetuning"] = adaptive
            self.stats["adaptive_finetuning"] = len(adaptive)
        
        # ============ Export All ============
        logger.info("\nExporting all data...")
        self.export_all_formats(all_prompts, dpo_pairs, ppo_trajectories, reward_data, curriculum, domain_prompts)
        
        # Save stats
        stats_file = OUTPUT_DIR / "ultimate_complete_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(dict(self.stats), f, indent=2, ensure_ascii=False)
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ ULTIMATE COMPLETE Prompt Generation Finished!")
        logger.info("=" * 80)
        
        for name, count in sorted(self.stats.items()):
            logger.info(f"  {name:30s}: {count:6d}")
        logger.info(f"  {'TOTAL':30s}: {sum(self.stats.values()):6d}")
    
    def export_all_formats(self, all_prompts: Dict, dpo_pairs: List, ppo_trajectories: List, 
                           reward_data: List, curriculum: Dict, domain_prompts: Dict):
        """Export all data in proper formats"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. SFT Data (OpenAI format)
        sft_path = OUTPUT_DIR / f"sft_data_{timestamp}.jsonl"
        with open(sft_path, 'w', encoding='utf-8') as f:
            for prompt_type, prompts in all_prompts.items():
                for p in prompts:
                    if "messages" in p:
                        json.dump({"messages": p["messages"]}, f, ensure_ascii=False)
                        f.write('\n')
        logger.info(f"SFT Data: {sft_path}")
        
        # 2. DPO Data (Preference pairs)
        dpo_path = OUTPUT_DIR / f"dpo_pairs_{timestamp}.jsonl"
        with open(dpo_path, 'w', encoding='utf-8') as f:
            for pair in dpo_pairs:
                json.dump({
                    "prompt": pair.prompt,
                    "chosen": pair.chosen,
                    "rejected": pair.rejected,
                    "metadata": {"domain": pair.domain, "strength": pair.preference_strength}
                }, f, ensure_ascii=False)
                f.write('\n')
        logger.info(f"DPO Data: {dpo_path}")
        
        # 3. ORPO/SimPO/CPO (Same format as DPO, different files)
        for variant in ["orpo", "simpo", "cpo"]:
            variant_path = OUTPUT_DIR / f"{variant}_pairs_{timestamp}.jsonl"
            with open(variant_path, 'w', encoding='utf-8') as f:
                for pair in dpo_pairs[:len(dpo_pairs)//2]:
                    json.dump({
                        "prompt": pair.prompt,
                        "chosen": pair.chosen,
                        "rejected": pair.rejected,
                    }, f, ensure_ascii=False)
                    f.write('\n')
            logger.info(f"{variant.upper()} Data: {variant_path}")
        
        # 4. PPO Data (Trajectories)
        ppo_path = OUTPUT_DIR / f"ppo_trajectories_{timestamp}.jsonl"
        with open(ppo_path, 'w', encoding='utf-8') as f:
            for traj in ppo_trajectories:
                json.dump({
                    "prompt": traj.prompt,
                    "responses": traj.responses,
                    "rewards": traj.rewards,
                    "final_response": traj.final_response,
                    "final_reward": traj.final_reward
                }, f, ensure_ascii=False)
                f.write('\n')
        logger.info(f"PPO Data: {ppo_path}")
        
        # 5. Reward Model Data
        reward_path = OUTPUT_DIR / f"reward_model_{timestamp}.jsonl"
        with open(reward_path, 'w', encoding='utf-8') as f:
            for sample in reward_data:
                json.dump(sample, f, ensure_ascii=False)
                f.write('\n')
        logger.info(f"Reward Model Data: {reward_path}")
        
        # 6. Curriculum Data (Stage-wise)
        for stage, examples in curriculum.items():
            curr_path = OUTPUT_DIR / f"curriculum_stage_{stage}_{timestamp}.jsonl"
            with open(curr_path, 'w', encoding='utf-8') as f:
                for ex in examples:
                    json.dump({
                        "messages": [
                            {"role": "system", "content": f"Curriculum Stage {stage}: {CURRICULUM_STAGES[stage]['description']}"},
                            {"role": "user", "content": ex.question},
                            {"role": "assistant", "content": ex.answer}
                        ],
                        "metadata": {"stage": stage, "topic": ex.topic, "domain": ex.domain}
                    }, f, ensure_ascii=False)
                    f.write('\n')
            logger.info(f"Curriculum Stage {stage}: {curr_path}")
        
        # 7. Domain-Specific SFT Splits
        for domain, prompts in domain_prompts.items():
            if prompts:
                domain_path = OUTPUT_DIR / domain / f"sft_{domain}_{timestamp}.jsonl"
                with open(domain_path, 'w', encoding='utf-8') as f:
                    for p in prompts:
                        if "messages" in p:
                            json.dump({"messages": p["messages"]}, f, ensure_ascii=False)
                            f.write('\n')
                logger.info(f"Domain {domain}: {domain_path} ({len(prompts)} examples)")
        
        # 8. Combined file
        combined_path = OUTPUT_DIR / f"ultimate_complete_{timestamp}.jsonl"
        logger.info(f"All data exported to {OUTPUT_DIR}")


def main():
    generator = UltimatePromptGenerator()
    generator.generate_all()

if __name__ == "__main__":
    main()