"""MambaGuard data pipeline: canonicalisation, graph builder, loaders.

Re-exports the public API used across the project:

* :class:`ProtocolMessage`     -- the canonical message record.
* :class:`MessageCanonicaliser`-- frozen sentence-transformer encoder.
* :class:`ProtocolGraph`       -- unified heterogeneous protocol graph G(t).
* :func:`temporal_split`       -- temporal (non-shuffling) split.
* :data:`DATASET_REGISTRY`     -- name -> loader callable mapping.
* :class:`ProtocolMessageDataset` and :func:`protocol_collate` --
  PyTorch ``Dataset`` and collate function. See ``protocol_collate``'s
  docstring for the dict-key contract honoured by the training/model code.
"""
from __future__ import annotations

from .canonicalize import (
    DEFAULT_METADATA_SCHEMA,
    EDGE_TYPES,
    MessageCanonicaliser,
    ProtocolMessage,
    from_a2a,
    from_acp,
    from_anp,
    from_mcp,
    metadata_to_vector,
)
from .datasets import (
    DATASET_REGISTRY,
    CompositeDataset,
    ProtocolMessageDataset,
    protocol_collate,
)
from .protocol_graph import NODE_TYPES, ProtocolGraph
from .splits import temporal_split

__all__ = [
    "ProtocolMessage",
    "MessageCanonicaliser",
    "metadata_to_vector",
    "DEFAULT_METADATA_SCHEMA",
    "EDGE_TYPES",
    "NODE_TYPES",
    "ProtocolGraph",
    "temporal_split",
    "DATASET_REGISTRY",
    "CompositeDataset",
    "ProtocolMessageDataset",
    "protocol_collate",
    "from_mcp",
    "from_acp",
    "from_a2a",
    "from_anp",
]
