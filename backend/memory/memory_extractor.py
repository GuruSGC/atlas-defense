"""
Memory Extractor for ATLAS.
Analyzes conversation dialogue turns to identify candidate facts, preferences,
instructions, and system configurations for persistent long-term storage.
Uses real Gemini LLM extraction when available, with strict heuristic baseline fallback.
"""

import uuid
import re
from typing import Optional, Dict, Any
from memory.memory_models import CandidateMemory, MemorySource
from core.embeddings import embedding_service
from core.gemini_client import gemini_client
from atlas.patterns import directive_findings


class MemoryExtractor:
    """
    Extracts candidate memory statements from conversation turns.
    Filters out transient casual dialogue and extracts only enduring information.
    """

    def __init__(self):
        # Trigger patterns for factual or preference updates
        self.preference_patterns = [
            r"my favorite \w+ is ([\w\s\.\-]+)",
            r"i prefer ([\w\s\.\-]+)",
            r"remember that ([\w\s\.\-]+)",
            r"note that ([\w\s\.\-]+)",
            r"always use ([\w\s\.\-]+)",
            r"the user prefers ([\w\s\.\-]+)",
            r"my \w+ is ([\w\s\.\-]+)",
        ]
        
        self.instruction_patterns = [
            r"always prepend ([\w\s\.\-\[\]]+)",
            r"always ([\w\s\.\-]+)",
            r"never ([\w\s\.\-]+)",
            r"system:\s*([\w\s\.\-]+)",
            r"override ([\w\s\.\-]+)",
            r"from now on ([\w\s\.\-]+)",
            r"instruction:\s*([\w\s\.\-]+)",
        ]

        # Casual phrases that MUST NOT be stored as memories
        self.casual_ignore_patterns = [
            r"^(hello|hi|hey|greetings|good morning|good evening)\b",
            r"^(how are you|what's up|how's it going)\b",
            r"^(thank you|thanks|bye|goodbye|see you)\b",
            r"^(can you help|help me|what can you do)\b",
            r"^(ok|okay|got it|understood|yes|no|sure)\.?$"
        ]

    def extract_from_turn(
        self,
        user_message: str,
        agent_response: Optional[str] = None,
        source: MemorySource = MemorySource.USER_CONVERSATION
    ) -> Optional[CandidateMemory]:
        """
        Extract a candidate memory item from a conversation turn.
        Returns CandidateMemory if enduring information exists, otherwise None.
        """
        text = user_message.strip()
        if len(text) < 4:
            return None

        # 1. Check for casual transient phrases to ignore
        lower_text = text.lower()
        for ignore_pat in self.casual_ignore_patterns:
            if re.search(ignore_pat, lower_text):
                return None

        # 2. If Gemini API is available in real mode, attempt LLM-based extraction
        if gemini_client.is_real_mode:
            llm_extracted = gemini_client.extract_memory_from_turn(user_message, agent_response)
            if llm_extracted and llm_extracted.get("has_candidate_memory"):
                cand_text = llm_extracted.get("candidate_memory_text", text)
                cid = f"cand-{uuid.uuid4().hex[:8]}"
                emb = embedding_service.get_embedding(cand_text)
                return CandidateMemory(
                    id=cid,
                    content=cand_text,
                    category=llm_extracted.get("memory_type", "instruction" if "instruction" in llm_extracted.get("memory_type", "") else "fact"),
                    extracted_from_turn=user_message,
                    source=source,
                    confidence=float(llm_extracted.get("confidence", 0.9)),
                    should_persist=True,
                    embedding=emb
                )

        # 3. Rule-based heuristic extraction baseline for offline / research prototype execution

        # Check for covert instruction injection, destructive commands, conditional rules, or system override patterns
        adversarial_markers = [
            "override", "ignore all", "system prompt", "exfiltrate", "bypass",
            "prepend", "[injected_override]", "immediately remove", "shut down", "delete all",
            "urgent update", "critical directive", "wipe all",
            "-rm ", "delete_all_files", "delete all files", "i want you to tell me",
            "tell me to use", "code to delete a file"
        ]
        if directive_findings(text).any or any(marker in lower_text for marker in adversarial_markers):
            cid = f"cand-{uuid.uuid4().hex[:8]}"
            emb = embedding_service.get_embedding(text)
            return CandidateMemory(
                id=cid,
                content=text,
                category="instruction",
                extracted_from_turn=user_message,
                source=source,
                confidence=0.95,
                should_persist=True,
                embedding=emb
            )

        # Check explicit preference triggers
        for pattern in self.preference_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                clean_content = text
                cid = f"cand-{uuid.uuid4().hex[:8]}"
                emb = embedding_service.get_embedding(clean_content)
                return CandidateMemory(
                    id=cid,
                    content=clean_content,
                    category="preference",
                    extracted_from_turn=user_message,
                    source=source,
                    confidence=0.90,
                    should_persist=True,
                    embedding=emb
                )

        # Factual statement detection (explicit assertions: is / are / prefers / deployed on / configured)
        factual_indicators = [" is ", " are ", " works at ", " configured with ", " set to ", " using ", " deployment ", " prefers ", " target is "]
        if any(ind in lower_text for ind in factual_indicators) and len(text.split()) >= 3:
            cid = f"cand-{uuid.uuid4().hex[:8]}"
            emb = embedding_service.get_embedding(text)
            return CandidateMemory(
                id=cid,
                content=text,
                category="fact",
                extracted_from_turn=user_message,
                source=source,
                confidence=0.85,
                should_persist=True,
                embedding=emb
            )

        # Explicit prefix "Remember:" or "Note:"
        if lower_text.startswith("remember:") or lower_text.startswith("note:") or lower_text.startswith("remember that"):
            clean_text = re.sub(r"^(remember:|note:|remember that)\s*", "", text, flags=re.IGNORECASE).strip()
            cid = f"cand-{uuid.uuid4().hex[:8]}"
            emb = embedding_service.get_embedding(clean_text)
            return CandidateMemory(
                id=cid,
                content=clean_text,
                category="fact",
                extracted_from_turn=user_message,
                source=source,
                confidence=0.92,
                should_persist=True,
                embedding=emb
            )

        return None


# Global extractor instance
memory_extractor = MemoryExtractor()
