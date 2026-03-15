"""
RAG Pipeline for Medical Chatbot

This module handles:
1. Loading the FAISS vector store
2. Retrieving relevant documents for queries
3. Generating responses using Gemini LLM
4. Calculating confidence scores
5. Using chat history for context when confidence is low
"""

import os
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from dotenv import load_dotenv
from langchain_core.documents import Document

# Load environment variables
load_dotenv()


@dataclass
class RAGResponse:
    """RAG response with confidence score."""
    answer: str
    confidence: float  # 0.0 to 1.0
    confidence_level: str  # "high", "medium", "low"
    retrieved_documents: List[Dict]
    prompt: str
    needs_more_info: bool


class MedicalRAGPipeline:
    """
    Retrieval-Augmented Generation pipeline for medical queries.
    Uses FAISS for document retrieval and Gemini for response generation.
    Includes confidence scoring and chat history context.
    """
    
    # Confidence thresholds
    HIGH_CONFIDENCE = 0.7
    MEDIUM_CONFIDENCE = 0.4
    
    def __init__(self, index_path: str = "vector_store/faiss_index"):
        """
        Initialize the RAG pipeline.
        
        Args:
            index_path: Path to the saved FAISS index
        """
        self.index_path = index_path
        self.db = None
        self.embeddings = None
        self.llm = None
        
        # Initialize components
        self._load_vector_store()
        self._initialize_llm()

    def _resolve_embedding_model(self) -> str:
        """Resolve embedding model name with persisted and env fallbacks."""
        env_model = os.getenv("EMBEDDING_MODEL")
        if env_model:
            return env_model

        model_path = os.path.join(self.index_path, "embedding_model.txt")
        if os.path.exists(model_path):
            try:
                with open(model_path, "r", encoding="utf-8") as f:
                    stored_model = f.read().strip()
                    if stored_model:
                        return stored_model
            except Exception as e:
                print(f"Warning: Unable to read saved embedding model: {e}")

        return "models/text-embedding-004"
    
    def _load_vector_store(self) -> None:
        """Load the FAISS vector store and embeddings."""
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            from langchain_community.vectorstores import FAISS
            
            model_name = self._resolve_embedding_model()
            print(f"Loading embeddings model ({model_name})...")
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model=model_name,
                task_type="retrieval_query"
            )
            
            print(f"Loading FAISS index from {self.index_path}...")
            if os.path.exists(self.index_path):
                self.db = FAISS.load_local(
                    self.index_path, 
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                print("FAISS index loaded successfully")
            else:
                print(f"Warning: FAISS index not found at {self.index_path}")
                print("Please run build_index.py first to create the index")
                
        except Exception as e:
            print(f"Error loading vector store: {e}")
            raise
    
    def _initialize_llm(self) -> None:
        """Initialize the Gemini LLM."""
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            
            print("Initializing Gemini LLM...")
            self.llm = ChatGoogleGenerativeAI(
                model="models/gemini-2.5-flash",
                temperature=0.3,
                max_tokens=1024,
                top_p=0.95,
                convert_system_message_to_human=True
            )
            print("Gemini LLM initialized successfully")
            
        except Exception as e:
            print(f"Error initializing LLM: {e}")
            raise
    
    def retrieve_documents(self, query: str, k: int = 3) -> Tuple[List[Dict], List[float]]:
        """
        Retrieve relevant documents from the vector store with similarity scores.
        
        Args:
            query: User query
            k: Number of documents to retrieve
            
        Returns:
            Tuple of (documents, similarity_scores)
        """
        if self.db is None:
            print("Warning: Vector store not loaded")
            return [], []
        
        try:
            # Use similarity_search_with_score to get scores
            results_with_scores = self.db.similarity_search_with_score(query, k=k)
            
            documents = []
            scores = []
            
            for doc, score in results_with_scores:
                documents.append({
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'source': doc.metadata.get('source', 'Unknown')
                })
                # FAISS returns L2 distance, convert to similarity (lower distance = higher similarity)
                # Normalize score to 0-1 range (assuming max distance ~2.0)
                similarity = max(0, 1 - (score / 2.0))
                scores.append(similarity)
            
            return documents, scores
            
        except Exception as e:
            print(f"Error retrieving documents: {e}")
            return [], []
    
    def calculate_confidence(self, 
                            query: str, 
                            documents: List[Dict], 
                            similarity_scores: List[float]) -> Tuple[float, str, bool]:
        """
        Calculate confidence score for the response.
        
        Args:
            query: User query
            documents: Retrieved documents
            similarity_scores: Similarity scores for documents
            
        Returns:
            Tuple of (confidence_score, confidence_level, needs_more_info)
        """
        if not similarity_scores:
            return 0.0, "low", True
        
        # Calculate average similarity score
        avg_similarity = np.mean(similarity_scores)
        max_similarity = max(similarity_scores) if similarity_scores else 0
        
        # Check if query has specific medical terms
        medical_terms = [
            "pain", "fever", "cough", "headache", "nausea", "vomiting", 
            "diarrhea", "rash", "chest", "heart", "breathing", "dizziness",
            "fatigue", "swelling", "bleeding", "infection", "symptoms"
        ]
        query_lower = query.lower()
        medical_term_count = sum(1 for term in medical_terms if term in query_lower)
        
        # Check query length (shorter queries are less specific)
        query_words = len(query.split())
        
        # Calculate confidence components
        similarity_weight = 0.5
        specificity_weight = 0.3
        term_weight = 0.2
        
        # Specificity score based on query length
        specificity_score = min(1.0, query_words / 10)  # Max at 10 words
        
        # Medical term score
        term_score = min(1.0, medical_term_count / 3)  # Max at 3 terms
        
        # Combined confidence score
        confidence = (
            similarity_weight * avg_similarity +
            specificity_weight * specificity_score +
            term_weight * term_score
        )
        
        # Adjust based on max similarity (if top result is very relevant, boost confidence)
        if max_similarity > 0.8:
            confidence = min(1.0, confidence * 1.1)
        
        # Determine confidence level
        if confidence >= self.HIGH_CONFIDENCE:
            level = "high"
            needs_more = False
        elif confidence >= self.MEDIUM_CONFIDENCE:
            level = "medium"
            needs_more = False
        else:
            level = "low"
            needs_more = True
        
        return confidence, level, needs_more
    
    def build_prompt(self, 
                     query: str, 
                     documents: List[Dict],
                     chat_history: Optional[List[Dict]] = None,
                     is_follow_up: bool = False) -> str:
        """
        Build the prompt for the LLM with context from retrieved documents.
        
        Args:
            query: User query
            documents: Retrieved documents
            chat_history: Previous chat messages for context
            is_follow_up: Whether this is a follow-up to a low-confidence query
            
        Returns:
            Formatted prompt string
        """
        # Build context from documents
        context_parts = []
        for i, doc in enumerate(documents, 1):
            context_parts.append(f"Document {i} (from {doc['source']}):\n{doc['content']}")
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Build chat history context if available
        history_context = ""
        if chat_history and len(chat_history) > 0:
            history_parts = []
            for msg in chat_history[-4:]:  # Last 4 messages for context
                role = "User" if msg["role"] == "user" else "Assistant"
                history_parts.append(f"{role}: {msg['content']}")
            history_context = "\n".join(history_parts)
        
        # Build the complete prompt
        if is_follow_up and history_context:
            prompt = f"""You are a helpful medical information assistant. Your role is to provide general health information based on the provided context and previous conversation.

IMPORTANT DISCLAIMERS:
- You are NOT a substitute for professional medical advice, diagnosis, or treatment
- Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition
- Never disregard professional medical advice or delay in seeking it because of information provided by this chatbot
- If you think you may have a medical emergency, call your doctor or emergency services immediately

PREVIOUS CONVERSATION:
{history_context}

MEDICAL CONTEXT:
{context}

USER'S FOLLOW-UP QUESTION: {query}

The user is providing additional information after being asked for more details. Please provide a helpful, accurate response based on all the context above. Be clear, concise, and empathetic. Include relevant warnings when appropriate."""
        else:
            prompt = f"""You are a helpful medical information assistant. Your role is to provide general health information based on the provided context. 

IMPORTANT DISCLAIMERS:
- You are NOT a substitute for professional medical advice, diagnosis, or treatment
- Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition
- Never disregard professional medical advice or delay in seeking it because of information provided by this chatbot
- If you think you may have a medical emergency, call your doctor or emergency services immediately

MEDICAL CONTEXT:
{context}

USER QUESTION: {query}

Please provide a helpful, accurate response based on the context above. Be clear, concise, and empathetic. Include relevant warnings when appropriate."""
        
        return prompt
    
    def generate_response(self, prompt: str) -> str:
        """
        Generate response using the Gemini LLM.
        
        Args:
            prompt: Formatted prompt
            
        Returns:
            Generated response text
        """
        try:
            from langchain_core.messages import HumanMessage
            
            messages = [HumanMessage(content=prompt)]
            response = self.llm.invoke(messages)
            
            return response.content
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return f"I apologize, but I encountered an error generating a response. Please try again. Error: {str(e)}"
    
    def get_rag_answer(self, 
                       query: str, 
                       chat_history: Optional[List[Dict]] = None,
                       k: int = 3) -> RAGResponse:
        """
        Complete RAG pipeline: retrieve documents, calculate confidence, and generate answer.
        
        Args:
            query: User query
            chat_history: Previous chat messages for context
            k: Number of documents to retrieve
            
        Returns:
            RAGResponse with answer, confidence, and metadata
        """
        # Step 1: Retrieve relevant documents with scores
        retrieved_docs, similarity_scores = self.retrieve_documents(query, k=k)
        
        # Step 2: Calculate confidence
        confidence, confidence_level, needs_more_info = self.calculate_confidence(
            query, retrieved_docs, similarity_scores
        )
        
        # Step 3: Build prompt with context
        is_follow_up = chat_history is not None and len(chat_history) > 0 and needs_more_info
        prompt = self.build_prompt(query, retrieved_docs, chat_history, is_follow_up)
        
        # Step 4: Generate response
        if needs_more_info and not is_follow_up:
            # Ask for more information instead of generating a medical response
            answer = self._generate_clarification_request(query)
        else:
            answer = self.generate_response(prompt)
        
        return RAGResponse(
            answer=answer,
            confidence=confidence,
            confidence_level=confidence_level,
            retrieved_documents=retrieved_docs,
            prompt=prompt,
            needs_more_info=needs_more_info and not is_follow_up
        )
    
    def _generate_clarification_request(self, query: str) -> str:
        """
        Generate a request for more information when confidence is low.
        
        Args:
            query: Original user query
            
        Returns:
            Clarification request message
        """
        return f"""I'd like to help you better, but I need more information about your symptoms. 

Could you please provide more details such as:
- When did the symptoms start?
- How severe are they (mild, moderate, severe)?
- Are there any other symptoms you're experiencing?
- Do you have any relevant medical history?
- Have you taken any medications for this?

The more details you share, the better I can assist you."""


# Singleton instance for reuse
_rag_pipeline = None


def get_rag_pipeline() -> MedicalRAGPipeline:
    """
    Get or create the RAG pipeline singleton.
    
    Returns:
        MedicalRAGPipeline instance
    """
    global _rag_pipeline
    if _rag_pipeline is None:
        _rag_pipeline = MedicalRAGPipeline()
    return _rag_pipeline


def get_rag_answer(query: str, 
                   chat_history: Optional[List[Dict]] = None,
                   k: int = 3) -> RAGResponse:
    """
    Convenience function to get RAG answer with confidence.
    
    Args:
        query: User query
        chat_history: Previous chat messages for context
        k: Number of documents to retrieve
        
    Returns:
        RAGResponse with answer and confidence
    """
    pipeline = get_rag_pipeline()
    return pipeline.get_rag_answer(query, chat_history=chat_history, k=k)


# Test function
if __name__ == "__main__":
    print("Testing RAG Pipeline with Confidence Scoring...")
    print("=" * 60)
    
    # Test queries with varying confidence levels
    test_queries = [
        "chest pain",  # Short, should be low confidence
        "What are the warning signs of a heart attack?",  # Detailed, should be high confidence
        "I have a severe headache and fever for 3 days, what should I do?",  # Detailed with symptoms
    ]
    
    pipeline = get_rag_pipeline()
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 60)
        
        result = pipeline.get_rag_answer(query)
        
        print(f"\nRetrieved {len(result.retrieved_documents)} documents")
        print(f"Confidence: {result.confidence:.2f} ({result.confidence_level})")
        print(f"Needs more info: {result.needs_more_info}")
        
        print(f"\nAnswer:\n{result.answer[:300]}...")
        print("=" * 60)
