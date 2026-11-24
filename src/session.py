"""Conversation history state."""
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
@dataclass
class HistoryEntry: question: str; answer: str; latency_ms: float; timestamp: str; settings: dict
class ConversationSession:
    def __init__(self): self.entries=[]
    def add(self, question, answer, latency_ms, settings):
        self.entries.append(HistoryEntry(question,answer,round(latency_ms,2),datetime.now(timezone.utc).isoformat(),settings)); return self.as_dicts()
    def as_dicts(self): return [asdict(x) for x in self.entries]
    def clear(self): self.entries.clear(); return []
