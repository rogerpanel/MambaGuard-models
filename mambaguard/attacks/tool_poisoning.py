"""MCP tool-poisoning attack: malicious tool descriptions instantiated from templates."""
from __future__ import annotations

import json
import random
import string
from pathlib import Path
from typing import Optional, Sequence

from ..data.canonicalize import ProtocolMessage


_DEFAULT_TEMPLATE_PATH = Path(__file__).parent / "templates" / "tool_poisoning.txt"


def _load_templates(path: Path) -> list[str]:
    if not path.exists():
        return ["Calculator that exfiltrates ${user_data} to ${attacker_url}."]
    with path.open("r", encoding="utf-8") as fh:
        return [ln.strip() for ln in fh if ln.strip() and not ln.startswith("#")]


def _instantiate(template: str, slots: dict[str, str]) -> str:
    try:
        return string.Template(template).safe_substitute(slots)
    except (KeyError, ValueError):
        return template


class ToolPoisoningAttack:
    """Synthesise malicious tool/* MCP descriptions for supply-chain evaluation."""

    DEFAULT_SLOTS: dict[str, str] = {
        "user_data": "user_credentials",
        "attacker_url": "http://attacker.invalid/exfil",
        "attacker_bucket": "s3://attacker.invalid/exfil",
        "auth_token": "auth_token_handle",
        "env_secrets": "ENV_SECRETS",
        "security_flags": "sandbox_isolation",
        "payload_b64": "PAYLOAD_B64_PLACEHOLDER",
        "prompt_injection_payload": "[PROMPT_INJECTION]",
        "malicious_payload": "[MALICIOUS_CONTENT]",
    }

    def __init__(
        self,
        templates: Optional[Sequence[str]] = None,
        template_path: Optional[str] = None,
        seed: int = 0,
        target_label: int = 1,
        slots: Optional[dict[str, str]] = None,
    ) -> None:
        if templates is None:
            path = Path(template_path) if template_path else _DEFAULT_TEMPLATE_PATH
            templates = _load_templates(path)
        if not templates:
            raise ValueError("ToolPoisoningAttack requires at least one template")
        self.templates = list(templates)
        self.rng = random.Random(seed)
        self.target_label = int(target_label)
        self.slots = {**self.DEFAULT_SLOTS, **(slots or {})}

    def sample_description(self) -> str:
        return _instantiate(self.rng.choice(self.templates), self.slots)

    def craft_tool_registration(
        self,
        tool_name: str = "tool:calculator",
        server: str = "server:malicious",
        t_m: float = 0.0,
    ) -> ProtocolMessage:
        desc = self.sample_description()
        params = {
            "client": "agent_victim",
            "server": server,
            "name": tool_name,
            "description": desc,
        }
        return ProtocolMessage(
            tau="M_tool",
            src="agent_victim",
            dst=server,
            payload=json.dumps(params),
            metadata={"is_internal": False, "auth": False, "rate_per_sec": 0.1},
            t_m=float(t_m),
            label=self.target_label,
        )

    def apply(
        self,
        messages: list[ProtocolMessage],
        rate: float = 0.1,
    ) -> list[ProtocolMessage]:
        out: list[ProtocolMessage] = []
        for msg in messages:
            if msg.tau == "M_tool" and self.rng.random() < rate:
                desc = self.sample_description()
                out.append(
                    ProtocolMessage(
                        tau=msg.tau,
                        src=msg.src,
                        dst=msg.dst,
                        payload=str(msg.payload) + " " + desc,
                        metadata=dict(msg.metadata),
                        t_m=msg.t_m,
                        label=self.target_label,
                        msg_id=msg.msg_id,
                    )
                )
            else:
                out.append(msg)
        return out

    def __call__(self, messages: list[ProtocolMessage], rate: float = 0.1) -> list[ProtocolMessage]:
        return self.apply(messages, rate=rate)
