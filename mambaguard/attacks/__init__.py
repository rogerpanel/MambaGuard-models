"""Adversarial attacks: gradient-based, noise-based and protocol-layer attacks."""
from __future__ import annotations

from .cw import CarliniWagnerL2
from .deepfool import DeepFool
from .fgsm import FGSM
from .mcp_attacks import MCPAdversaryBench, MCPTestCase
from .noise import GaussianNoise, LabelMasking
from .pgd import PGD
from .prompt_injection import PromptInjectionAttack
from .tool_poisoning import ToolPoisoningAttack

__all__ = [
    "FGSM",
    "PGD",
    "CarliniWagnerL2",
    "DeepFool",
    "GaussianNoise",
    "LabelMasking",
    "PromptInjectionAttack",
    "ToolPoisoningAttack",
    "MCPAdversaryBench",
    "MCPTestCase",
]
