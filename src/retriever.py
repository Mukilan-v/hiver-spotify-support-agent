"""
Resolution Retriever (RAG) Module for Spotify Customer Support.
Indexes verified support playbooks and historical resolutions.
Provides intent-filtered retrieval of resolutions, canonical links, and suggested actions.
"""

import json
import os
import re
from typing import Dict, Any, List, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def clean_text(text: str) -> str:
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    text = re.sub(r'https?://\S+', '', text)
    return re.sub(r'\s+', ' ', text).strip()

class ResolutionRetriever:
    def __init__(self, kb_path: Optional[str] = None):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if not kb_path:
            kb_path = os.path.join(base_dir, "data", "kb", "resolution_kb.json")
        
        self.kb_path = kb_path
        self.documents: List[Dict[str, Any]] = []
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        self.doc_vectors = None
        self._load_and_index()

    def _load_and_index(self):
        if not os.path.exists(self.kb_path):
            raise FileNotFoundError(f"Knowledge base file not found at {self.kb_path}")
        
        with open(self.kb_path, "r", encoding="utf-8") as f:
            self.documents = json.load(f)

        # Build corpus texts combining title, sub_intent, key actions, and canonical response
        corpus = []
        for doc in self.documents:
            text_repr = f"{doc.get('title', '')} {doc.get('sub_intent', '')} {' '.join(doc.get('key_actions', []))} {doc.get('canonical_response', '')}"
            corpus.append(clean_text(text_repr))

        self.doc_vectors = self.vectorizer.fit_transform(corpus)

    def retrieve(self, query: str, intent: Optional[str] = None, top_k: int = 2) -> List[Dict[str, Any]]:
        """
        Retrieve the top-k most relevant playbooks for a query.
        Optionally boosts or filters by classified intent.
        """
        cleaned_query = clean_text(query)
        if not cleaned_query:
            return self.documents[:top_k]

        query_vec = self.vectorizer.transform([cleaned_query])
        similarities = cosine_similarity(query_vec, self.doc_vectors)[0]

        # Intent boost: boost scores matching the predicted intent
        if intent:
            for i, doc in enumerate(self.documents):
                if doc.get("intent") == intent:
                    similarities[i] += 0.35

        # Rank indices
        ranked_indices = np.argsort(similarities)[::-1]
        results = []
        for idx in ranked_indices[:top_k]:
            doc_copy = dict(self.documents[idx])
            doc_copy["retrieval_score"] = round(float(similarities[idx]), 4)
            results.append(doc_copy)

        return results
