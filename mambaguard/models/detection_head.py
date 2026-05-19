"""Detection head over the 34-class RobustIDPS.ai attack taxonomy."""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# RobustIDPS.ai 34-class attack taxonomy grouped by family.
# Severity tiers follow the paper's harm-impact mapping (low / medium / high / critical).
_LABELS: List[str] = [
    # injection family (0-7)
    "benign",
    "direct_prompt_injection",
    "indirect_prompt_injection",
    "system_prompt_leak",
    "jailbreak_roleplay",
    "jailbreak_obfuscated",
    "tool_argument_injection",
    "code_injection",
    # exfiltration family (8-13)
    "memory_exfiltration",
    "credential_exfiltration",
    "pii_exfiltration",
    "model_weight_probing",
    "context_window_exfiltration",
    "side_channel_timing",
    # poisoning family (14-19)
    "training_data_poisoning",
    "rag_corpus_poisoning",
    "tool_response_poisoning",
    "vector_store_poisoning",
    "feedback_loop_poisoning",
    "backdoor_trigger",
    # protocol-abuse family (20-25)
    "mcp_handshake_spoof",
    "a2a_capability_forgery",
    "acp_replay",
    "anp_route_hijack",
    "tool_schema_tampering",
    "auth_token_replay",
    # availability family (26-29)
    "infinite_loop_invocation",
    "token_flood",
    "tool_storm",
    "context_bomb",
    # social-engineering family (30-33)
    "agent_impersonation",
    "human_in_the_loop_phish",
    "consensus_manipulation",
    "delegation_abuse",
]

_FAMILY_SEVERITY: Dict[str, str] = {
    "benign": "low",
    "injection": "high",
    "exfiltration": "critical",
    "poisoning": "critical",
    "protocol": "high",
    "availability": "medium",
    "social": "high",
}


def _family_of(label: str) -> str:
    if label == "benign":
        return "benign"
    if "injection" in label or "jailbreak" in label or "leak" in label:
        return "injection"
    if "exfiltration" in label or "probing" in label or "side_channel" in label:
        return "exfiltration"
    if "poisoning" in label or "backdoor" in label:
        return "poisoning"
    if any(k in label for k in ("mcp", "a2a", "acp", "anp", "schema", "auth")):
        return "protocol"
    if any(k in label for k in ("loop", "flood", "storm", "bomb")):
        return "availability"
    return "social"


class DetectionHead(nn.Module):
    """Two-layer MLP classifier mapping fused agent features to 34 attack-class logits."""

    LABELS: List[str] = _LABELS

    def __init__(
        self,
        d_in: int,
        num_classes: int = 34,
        hidden: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_classes != len(_LABELS):
            # Allow override but warn via assertion only when paper-default.
            pass
        self.d_in = d_in
        self.num_classes = num_classes
        hidden = hidden or d_in
        self.norm = nn.LayerNorm(d_in)
        self.fc1 = nn.Linear(d_in, hidden)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: (..., d_in). Returns (..., num_classes) raw logits."""
        h = self.norm(x)
        h = F.gelu(self.fc1(h))
        h = self.drop(h)
        return self.fc2(h)

    @classmethod
    def severity_for(cls, label: str | int) -> str:
        """Return one of {low, medium, high, critical} for the given label or index."""
        if isinstance(label, int):
            if not 0 <= label < len(_LABELS):
                raise IndexError(f"label index {label} out of range")
            label = _LABELS[label]
        if label not in _LABELS:
            raise KeyError(f"unknown label {label!r}")
        return _FAMILY_SEVERITY[_family_of(label)]
