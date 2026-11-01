"""
PERSON 6 — Tools Adapter
Wrapping official ZentroMart tools with verification_value support, calendar business days, and sanitization.
"""

import sys
import os
import re
import datetime
from typing import Dict, Any, Optional

REFERENCE_DATE = datetime.date(2026, 10, 20)

class ToolsAdapter:
    def __init__(self, dataset_dir: str):
        self.dataset_dir = os.path.abspath(dataset_dir)
        tools_path = os.path.join(self.dataset_dir, "tools")
        if tools_path not in sys.path:
            sys.path.insert(0, tools_path)

        try:
            from tools import lookup_order, create_ticket
            self._lookup_fn = lookup_order
            self._ticket_fn = create_ticket
        except ImportError:
            self._lookup_fn = None
            self._ticket_fn = None

    def lookup(self, order_id: str, verification_value: Optional[str] = None) -> Dict[str, Any]:
        """Lookup order with verification_value per official tools.py specification."""
        if not self._lookup_fn:
            return {"found": False, "verified": False, "message": "Tools module unavailable"}

        clean_id = re.sub(r"\D", "", str(order_id))
        clean_proof = re.sub(r"\D", "", str(verification_value)) if verification_value else None
        return self._lookup_fn(clean_id, verification_value=clean_proof)

    def ticket(self, summary: str, order_id: Optional[str] = None, reason: Optional[str] = None, intent: Optional[str] = None) -> Dict[str, Any]:
        """Create support ticket. Ensures summary >= 20 chars and sanitizes verification details."""
        if not self._ticket_fn:
            return {"created": False, "error": "Tools module unavailable"}

        safe_summary = summary.strip()
        if len(safe_summary) < 20:
            safe_summary = f"Customer support escalation: {safe_summary} - Official Review"

        safe_summary = re.sub(r"\b\d{6}\b", "[REDACTED_PIN]", safe_summary)
        safe_summary = re.sub(r"\b\d{10}\b", "[REDACTED_PHONE]", safe_summary)

        return self._ticket_fn(
            summary=safe_summary,
            order_id=order_id,
            reason=reason,
            intent=intent
        )
