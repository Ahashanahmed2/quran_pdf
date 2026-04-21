# RLHF_chat_system.py
#!/usr/bin/env python3
"""
ChatGPT-Style RLHF Training System
Human-in-the-Loop Learning for Tafsir AI
"""

import os
import json
import hashlib
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum

from pinecone import Pinecone
from pymongo import MongoClient
from openai import OpenAI

# ============ Configuration ============
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "islamic-knowledge")
MONGODB_URI = os.environ.get("MONGODB_URI")
MONGODB_DB = os.environ.get("MONGODB_DB", "islamic_library")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# ============ Enums ============

class FeedbackRating(int, Enum):
    """Human feedback rating"""
    VERY_BAD = 1
    BAD = 2
    AVERAGE = 3
    GOOD = 4
    EXCELLENT = 5

class FeedbackAction(str, Enum):
    """Action after feedback"""
    ACCEPT = "accept"
    CORRECT = "correct"
    REGENERATE = "regenerate"
    REJECT = "reject"

class TrainingStatus(str, Enum):
    """Training data status"""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    USED = "used"
    ARCHIVED = "archived"

# ============ Data Models ============

@dataclass
class ChatMessage:
    """Single chat message"""
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self):
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat()
        }

@dataclass
class RetrievedContext:
    """Context retrieved from Pinecone"""
    chunk_id: str
    text: str
    book_name: str
    volume: int
    page_number: int
    similarity_score: float
    content_type: str
    namespace: str

@dataclass
class AIResponse:
    """Complete AI response with metadata"""
    question: str
    answer: str
    contexts: List[RetrievedContext]
    references: List[Dict]
    confidence: float
    generated_at: datetime = field(default_factory=datetime.utcnow)
    model_used: str = "gpt-4o"
    response_id: str = field(default_factory=lambda: hashlib.md5(str(datetime.utcnow().timestamp()).encode()).hexdigest()[:12])

@dataclass
class HumanFeedback:
    """Human feedback on AI response"""
    response_id: str
    question: str
    ai_answer: str
    rating: FeedbackRating
    action: FeedbackAction
    corrected_answer: Optional[str] = None
    feedback_notes: Optional[str] = None
    is_accurate: bool = False
    is_helpful: bool = False
    is_complete: bool = False
    missing_info: Optional[List[str]] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self):
        return {
            "response_id": self.response_id,
            "question": self.question,
            "ai_answer": self.ai_answer,
            "rating": self.rating.value,
            "action": self.action.value,
            "corrected_answer": self.corrected_answer,
            "feedback_notes": self.feedback_notes,
            "is_accurate": self.is_accurate,
            "is_helpful": self.is_helpful,
            "is_complete": self.is_complete,
            "missing_info": self.missing_info,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "timestamp": self.timestamp.isoformat()
        }

@dataclass
class TrainingExample:
    """Training example for fine-tuning"""
    conversation_id: str
    messages: List[Dict]  # [{"role": "user", "content": "..."}, ...]
    quality_score: float
    source: str  # "human_feedback", "curated", "synthetic"
    metadata: Dict = field(default_factory=dict)
    status: TrainingStatus = TrainingStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    used_at: Optional[datetime] = None

# ============ Main RLHF Chat System ============

class TafsirChatRLHFSystem:
    """
    ChatGPT-Style RLHF Training System
    """
    
    def __init__(self):
        # Pinecone
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.index = self.pc.Index(PINECONE_INDEX_NAME)
        
        # MongoDB
        self.mongo_client = MongoClient(MONGODB_URI)
        self.mongo_db = self.mongo_client[MONGODB_DB]
        self._init_collections()
        
        # OpenAI
        self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
        
        # Session tracking
        self.current_session_id = None
        self.conversation_history = []
    
    def _init_collections(self):
        """Initialize MongoDB collections"""
        collections = [
            "chat_sessions",
            "chat_messages",
            "ai_responses",
            "human_feedback",
            "training_examples",
            "dpo_preference_pairs",
            "fine_tuning_jobs"
        ]
        for coll in collections:
            if coll not in self.mongo_db.list_collection_names():
                self.mongo_db.create_collection(coll)
    
    def start_new_session(self, user_id: Optional[str] = None) -> str:
        """Start a new chat session"""
        self.current_session_id = hashlib.md5(f"{user_id}_{datetime.utcnow().timestamp()}".encode()).hexdigest()[:16]
        self.conversation_history = []
        
        self.mongo_db["chat_sessions"].insert_one({
            "session_id": self.current_session_id,
            "user_id": user_id,
            "started_at": datetime.utcnow(),
            "status": "active"
        })
        
        return self.current_session_id
    
    def semantic_search(self, query: str, namespace: str = None, top_k: int = 10) -> List[RetrievedContext]:
        """
        Pinecone-এ সার্চ করে প্রাসঙ্গিক কন্টেক্সট আনে
        """
        # Generate embedding
        embedding_response = self.openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=query
        )
        query_embedding = embedding_response.data[0].embedding
        
        # Search in specified namespace or all
        namespaces = [namespace] if namespace else ["tafsir", "hadith", "fiqh", "seerah", "general"]
        
        all_results = []
        for ns in namespaces:
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k // len(namespaces) + 1,
                include_metadata=True,
                namespace=ns
            )
            
            for match in results.matches:
                all_results.append(RetrievedContext(
                    chunk_id=match.id,
                    text=match.metadata.get("text_preview", ""),
                    book_name=match.metadata.get("book_name", ""),
                    volume=match.metadata.get("volume", 0),
                    page_number=match.metadata.get("page_number", 0),
                    similarity_score=match.score,
                    content_type=match.metadata.get("content_type", "unknown"),
                    namespace=ns
                ))
        
        # Sort by similarity and deduplicate
        all_results.sort(key=lambda x: x.similarity_score, reverse=True)
        
        # Remove duplicates (same chunk)
        seen = set()
        unique_results = []
        for ctx in all_results:
            if ctx.chunk_id not in seen:
                seen.add(ctx.chunk_id)
                unique_results.append(ctx)
        
        return unique_results[:top_k]
    
    def get_full_text(self, chunk_id: str) -> Optional[str]:
        """MongoDB থেকে সম্পূর্ণ টেক্সট আনুন"""
        doc = self.mongo_db["chunks"].find_one({"chunk_id": chunk_id})
        return doc.get("text") if doc else None
    
    def generate_response(self, question: str, namespace: str = "tafsir") -> AIResponse:
        """
        প্রশ্নের উত্তর জেনারেট করুন
        """
        # Step 1: Retrieve context
        contexts = self.semantic_search(question, namespace=namespace, top_k=5)
        
        if not contexts:
            return AIResponse(
                question=question,
                answer="দুঃখিত, এই প্রশ্নের উত্তর দেওয়ার মতো কোনো তথ্য পাওয়া যায়নি।",
                contexts=[],
                references=[],
                confidence=0.0
            )
        
        # Step 2: Get full text for top contexts
        full_contexts = []
        for ctx in contexts[:3]:
            full_text = self.get_full_text(ctx.chunk_id)
            if full_text:
                ctx.text = full_text
            full_contexts.append(ctx)
        
        # Step 3: Build context for LLM
        context_text = "\n\n".join([
            f"[সোর্স {i+1}]\n📚 {ctx.book_name}, খন্ড {ctx.volume}, পৃষ্ঠা {ctx.page_number}\n{ctx.text[:1000]}"
            for i, ctx in enumerate(full_contexts)
        ])
        
        # Step 4: Generate response
        system_prompt = """আপনি একজন ইসলামিক স্কলার এবং তাফসীর বিশেষজ্ঞ। আপনার কাজ হল প্রশ্নের সঠিক উত্তর দেওয়া।

নির্দেশাবলী:
1. প্রদত্ত সোর্স থেকে তথ্য নিন
2. প্রতিটি দাবির জন্য সোর্স উল্লেখ করুন [সোর্স X]
3. উত্তর বাংলা ভাষায় দিন
4. প্রয়োজনে টেবিল বা তালিকা ব্যবহার করুন
5. সোর্সে তথ্য না থাকলে বলুন "এই বিষয়ে সোর্সে তথ্য নেই"
6. উত্তর শেষে রেফারেন্স সেকশন যোগ করুন"""

        user_prompt = f"""প্রশ্ন: {question}

প্রসঙ্গ (Context):
{context_text}

উপরের প্রসঙ্গের ভিত্তিতে প্রশ্নের উত্তর দিন।"""

        response = self.openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=1500
        )
        
        answer = response.choices[0].message.content
        
        # Step 5: Build references
        references = []
        for i, ctx in enumerate(full_contexts):
            references.append({
                "source_number": i + 1,
                "book_name": ctx.book_name,
                "volume": ctx.volume,
                "page_number": ctx.page_number,
                "content_type": ctx.content_type,
                "similarity": round(ctx.similarity_score, 3)
            })
        
        # Step 6: Calculate confidence
        avg_similarity = sum(c.similarity_score for c in contexts) / len(contexts) if contexts else 0
        confidence = avg_similarity * 0.7 + 0.3  # Adjust based on context quality
        
        ai_response = AIResponse(
            question=question,
            answer=answer,
            contexts=full_contexts,
            references=references,
            confidence=round(confidence, 3)
        )
        
        # Save to MongoDB
        self.mongo_db["ai_responses"].insert_one({
            **asdict(ai_response),
            "contexts": [asdict(c) for c in full_contexts],
            "session_id": self.current_session_id
        })
        
        # Update conversation history
        self.conversation_history.append(ChatMessage(role="user", content=question))
        self.conversation_history.append(ChatMessage(role="assistant", content=answer))
        
        return ai_response
    
    def collect_feedback(self, response_id: str, rating: int, action: str,
                         corrected_answer: Optional[str] = None,
                         feedback_notes: Optional[str] = None,
                         is_accurate: bool = False,
                         is_helpful: bool = False,
                         is_complete: bool = False,
                         missing_info: Optional[List[str]] = None) -> HumanFeedback:
        """
        মানুষের ফিডব্যাক সংগ্রহ করুন
        """
        # Get original response
        original = self.mongo_db["ai_responses"].find_one({"response_id": response_id})
        
        if not original:
            raise ValueError(f"Response {response_id} not found")
        
        feedback = HumanFeedback(
            response_id=response_id,
            question=original["question"],
            ai_answer=original["answer"],
            rating=FeedbackRating(rating),
            action=FeedbackAction(action),
            corrected_answer=corrected_answer,
            feedback_notes=feedback_notes,
            is_accurate=is_accurate,
            is_helpful=is_helpful,
            is_complete=is_complete,
            missing_info=missing_info,
            session_id=self.current_session_id
        )
        
        # Save feedback
        self.mongo_db["human_feedback"].insert_one(feedback.to_dict())
        
        # Generate training example if good feedback
        if rating >= 4 or (corrected_answer and len(corrected_answer) > 50):
            self._create_training_example_from_feedback(feedback, original)
        
        return feedback
    
    def _create_training_example_from_feedback(self, feedback: HumanFeedback, original_response: Dict):
        """
        ফিডব্যাক থেকে ট্রেনিং উদাহরণ তৈরি করুন
        """
        messages = []
        
        # System message
        messages.append({
            "role": "system",
            "content": "আপনি একজন ইসলামিক স্কলার এবং তাফসীর বিশেষজ্ঞ। প্রশ্নের সঠিক ও সোর্স-সমর্থিত উত্তর দিন।"
        })
        
        # User question
        messages.append({
            "role": "user",
            "content": feedback.question
        })
        
        # Assistant answer (use corrected if available)
        final_answer = feedback.corrected_answer if feedback.corrected_answer else original_response["answer"]
        messages.append({
            "role": "assistant",
            "content": final_answer
        })
        
        # Determine quality score
        if feedback.rating == FeedbackRating.EXCELLENT:
            quality_score = 1.0
        elif feedback.rating == FeedbackRating.GOOD:
            quality_score = 0.85
        elif feedback.rating == FeedbackRating.AVERAGE:
            quality_score = 0.7
        else:
            quality_score = 0.5
        
        # Create training example
        example = TrainingExample(
            conversation_id=hashlib.md5(f"{feedback.response_id}_training".encode()).hexdigest()[:16],
            messages=messages,
            quality_score=quality_score,
            source="human_feedback",
            metadata={
                "response_id": feedback.response_id,
                "original_answer": original_response["answer"],
                "rating": feedback.rating.value,
                "action": feedback.action.value,
                "feedback_notes": feedback.feedback_notes,
                "references": original_response.get("references", [])
            },
            status=TrainingStatus.APPROVED if quality_score >= 0.7 else TrainingStatus.PENDING
        )
        
        self.mongo_db["training_examples"].insert_one(asdict(example))
        
        # Create DPO preference pair if we have both good and bad answers
        if feedback.action == FeedbackAction.CORRECT and feedback.corrected_answer:
            self._create_dpo_pair(feedback, original_response)
    
    def _create_dpo_pair(self, feedback: HumanFeedback, original_response: Dict):
        """
        DPO (Direct Preference Optimization) এর জন্য পছন্দের জোড়া তৈরি করুন
        """
        dpo_pair = {
            "pair_id": hashlib.md5(f"{feedback.response_id}_dpo".encode()).hexdigest()[:16],
            "question": feedback.question,
            "chosen": feedback.corrected_answer,  # Human corrected (preferred)
            "rejected": original_response["answer"],  # Original AI (less preferred)
            "metadata": {
                "response_id": feedback.response_id,
                "rating": feedback.rating.value,
                "feedback_notes": feedback.feedback_notes
            },
            "created_at": datetime.utcnow(),
            "status": "pending"
        }
        
        self.mongo_db["dpo_preference_pairs"].insert_one(dpo_pair)
    
    def get_training_data(self, status: TrainingStatus = TrainingStatus.APPROVED, 
                          limit: int = 100) -> List[Dict]:
        """
        ফাইন-টিউনিংয়ের জন্য ট্রেনিং ডেটা পান
        """
        examples = self.mongo_db["training_examples"].find({
            "status": status.value,
            "quality_score": {"$gte": 0.7}
        }).limit(limit)
        
        return list(examples)
    
    def export_training_data_jsonl(self, filepath: str, status: TrainingStatus = TrainingStatus.APPROVED):
        """
        JSONL ফরম্যাটে ট্রেনিং ডেটা এক্সপোর্ট করুন
        """
        examples = self.get_training_data(status=status, limit=10000)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            for ex in examples:
                json.dump({"messages": ex["messages"]}, f, ensure_ascii=False)
                f.write('\n')
        
        print(f"[Export] {len(examples)} examples exported to {filepath}")
        
        # Mark as used
        for ex in examples:
            self.mongo_db["training_examples"].update_one(
                {"conversation_id": ex["conversation_id"]},
                {"$set": {"status": TrainingStatus.USED.value, "used_at": datetime.utcnow()}}
            )
    
    def export_dpo_data_jsonl(self, filepath: str):
        """
        DPO ট্রেনিংয়ের জন্য ডেটা এক্সপোর্ট করুন
        """
        pairs = self.mongo_db["dpo_preference_pairs"].find({"status": "pending"}).limit(10000)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            for pair in pairs:
                json.dump({
                    "question": pair["question"],
                    "chosen": pair["chosen"],
                    "rejected": pair["rejected"]
                }, f, ensure_ascii=False)
                f.write('\n')
        
        count = self.mongo_db["dpo_preference_pairs"].count_documents({"status": "pending"})
        print(f"[Export] {count} DPO pairs exported to {filepath}")
        
        # Mark as used
        self.mongo_db["dpo_preference_pairs"].update_many(
            {"status": "pending"},
            {"$set": {"status": "used", "used_at": datetime.utcnow()}}
        )
    
    def get_statistics(self) -> Dict:
        """
        সিস্টেম পরিসংখ্যান
        """
        return {
            "total_sessions": self.mongo_db["chat_sessions"].count_documents({}),
            "total_responses": self.mongo_db["ai_responses"].count_documents({}),
            "total_feedback": self.mongo_db["human_feedback"].count_documents({}),
            "training_examples": {
                "pending": self.mongo_db["training_examples"].count_documents({"status": "pending"}),
                "approved": self.mongo_db["training_examples"].count_documents({"status": "approved"}),
                "used": self.mongo_db["training_examples"].count_documents({"status": "used"})
            },
            "dpo_pairs": {
                "pending": self.mongo_db["dpo_preference_pairs"].count_documents({"status": "pending"}),
                "used": self.mongo_db["dpo_preference_pairs"].count_documents({"status": "used"})
            },
            "average_rating": self._calculate_average_rating(),
            "feedback_by_action": self._get_feedback_by_action()
        }
    
    def _calculate_average_rating(self) -> float:
        """গড় রেটিং ক্যালকুলেট করুন"""
        pipeline = [
            {"$group": {"_id": None, "avg": {"$avg": "$rating"}}}
        ]
        result = list(self.mongo_db["human_feedback"].aggregate(pipeline))
        return round(result[0]["avg"], 2) if result else 0.0
    
    def _get_feedback_by_action(self) -> Dict:
        """অ্যাকশন অনুযায়ী ফিডব্যাক গণনা"""
        pipeline = [
            {"$group": {"_id": "$action", "count": {"$sum": 1}}}
        ]
        results = list(self.mongo_db["human_feedback"].aggregate(pipeline))
        return {r["_id"]: r["count"] for r in results if r["_id"]}

# ============ FastAPI Chat Interface ============

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

app = FastAPI(title="Tafsir Chat RLHF API", version="1.0.0")
chat_system = TafsirChatRLHFSystem()

class QuestionRequest(BaseModel):
    question: str
    namespace: str = "tafsir"
    session_id: Optional[str] = None

class FeedbackRequest(BaseModel):
    response_id: str
    rating: int  # 1-5
    action: str  # accept, correct, regenerate, reject
    corrected_answer: Optional[str] = None
    feedback_notes: Optional[str] = None
    is_accurate: bool = False
    is_helpful: bool = False
    is_complete: bool = False
    missing_info: Optional[List[str]] = None

@app.post("/api/chat/ask")
async def ask_question(request: QuestionRequest):
    """প্রশ্ন জিজ্ঞাসা করুন এবং AI উত্তর পান"""
    if request.session_id:
        chat_system.current_session_id = request.session_id
    else:
        chat_system.start_new_session()
    
    response = chat_system.generate_response(request.question, namespace=request.namespace)
    
    return JSONResponse({
        "session_id": chat_system.current_session_id,
        "response_id": response.response_id,
        "question": response.question,
        "answer": response.answer,
        "references": response.references,
        "confidence": response.confidence
    })

@app.post("/api/chat/feedback")
async def submit_feedback(request: FeedbackRequest):
    """AI উত্তরের উপর ফিডব্যাক দিন"""
    try:
        feedback = chat_system.collect_feedback(
            response_id=request.response_id,
            rating=request.rating,
            action=request.action,
            corrected_answer=request.corrected_answer,
            feedback_notes=request.feedback_notes,
            is_accurate=request.is_accurate,
            is_helpful=request.is_helpful,
            is_complete=request.is_complete,
            missing_info=request.missing_info
        )
        
        return JSONResponse({
            "success": True,
            "message": "আপনার ফিডব্যাকের জন্য ধন্যবাদ! AI এটি থেকে শিখবে।",
            "feedback_id": feedback.response_id
        })
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/admin/statistics")
async def get_statistics():
    """অ্যাডমিন পরিসংখ্যান"""
    return chat_system.get_statistics()

@app.post("/api/admin/export/training")
async def export_training_data():
    """ট্রেনিং ডেটা এক্সপোর্ট করুন"""
    chat_system.export_training_data_jsonl("/tmp/training_data.jsonl")
    chat_system.export_dpo_data_jsonl("/tmp/dpo_data.jsonl")
    return {"success": True, "message": "Training data exported"}

# ============ Usage Example ============
if __name__ == "__main__":
    import uvicorn
    
    # Start chat session
    session_id = chat_system.start_new_session(user_id="test_user")
    print(f"Session started: {session_id}")
    
    # Ask question
    response = chat_system.generate_response(
        "সূরা ফাতিহার প্রথম আয়াতের তাফসীর কী?",
        namespace="tafsir"
    )
    
    print(f"\n📝 প্রশ্ন: {response.question}")
    print(f"\n🤖 উত্তর:\n{response.answer}")
    print(f"\n📚 রেফারেন্স:")
    for ref in response.references:
        print(f"  - {ref['book_name']}, খন্ড {ref['volume']}, পৃষ্ঠা {ref['page_number']}")
    
    # Submit feedback
    feedback = chat_system.collect_feedback(
        response_id=response.response_id,
        rating=5,
        action="accept",
        is_accurate=True,
        is_helpful=True,
        is_complete=True,
        feedback_notes="চমৎকার উত্তর, সব সোর্স সঠিক"
    )
    
    print(f"\n✅ ফিডব্যাক জমা হয়েছে")
    
    # Show statistics
    stats = chat_system.get_statistics()
    print(f"\n📊 পরিসংখ্যান:")
    print(f"  - মোট রেসপন্স: {stats['total_responses']}")
    print(f"  - মোট ফিডব্যাক: {stats['total_feedback']}")
    print(f"  - গড় রেটিং: {stats['average_rating']}")
    print(f"  - ট্রেনিং উদাহরণ: {stats['training_examples']['approved']}")
    
    # Run API server
    # uvicorn.run(app, host="0.0.0.0", port=8000)