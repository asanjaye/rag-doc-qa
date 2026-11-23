import json
import httpx
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class EscalationRequest:
    question: str
    reason: str
    timestamp: str
    retrieval_scores: list[float]
    top_sources: list[str]
    request_id: str
    
    def to_dict(self) -> dict:
        return asdict(self)


class EscalationHandler:
    def __init__(
        self,
        webhook_url: str = "",
        log_file: str = "escalations.jsonl"
    ):
        self.webhook_url = webhook_url
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
    
    def escalate(self, request: EscalationRequest) -> bool:
        self._log_to_file(request)
        
        webhook_success = True
        if self.webhook_url:
            webhook_success = self._send_webhook(request)
        
        return webhook_success
    
    def _log_to_file(self, request: EscalationRequest):
        with open(self.log_file, "a") as f:
            f.write(json.dumps(request.to_dict()) + "\n")
        logger.info(f"Escalation logged: {request.request_id}")
    
    def _send_webhook(self, request: EscalationRequest) -> bool:
        payload = {
            "event": "human_handoff_required",
            "data": request.to_dict(),
            "source": "document-qa-assistant"
        }
        
        try:
            response = httpx.post(
                self.webhook_url,
                json=payload,
                timeout=10.0
            )
            response.raise_for_status()
            logger.info(f"Webhook sent successfully for {request.request_id}")
            return True
        except Exception as e:
            logger.error(f"Webhook failed for {request.request_id}: {e}")
            return False
    
    def get_pending_escalations(self, limit: int = 50) -> list[EscalationRequest]:
        if not self.log_file.exists():
            return []
        
        escalations = []
        with open(self.log_file) as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    escalations.append(EscalationRequest(**data))
        
        return escalations[-limit:]
    
    def clear_escalations(self):
        if self.log_file.exists():
            self.log_file.unlink()
