"""Unified Heterogeneous Protocol Graph G(t)=(V(t), E(t), X(t)).

Implements Section 3.2 of the MambaGuard paper: a typed temporal multigraph
whose vertex partitions are agents (V_A), tools/services (V_T), capability
descriptors (V_C) and active sessions (V_S). Edges are typed by the channel
tag ``tau`` of the producing :class:`ProtocolMessage` and carry the
concatenation ``[payload_emb (d_p) || metadata_vec (d_mu)]`` as edge features.

The graph builder is dependency-light: it stores native Python structures and
only constructs a ``torch_geometric.data.Data`` object on demand. If PyG is
unavailable a structurally-equivalent ``dict`` is returned so downstream code
can still tensorise.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final, Generator, Iterable, Optional

from .canonicalize import (
    DEFAULT_METADATA_SCHEMA,
    EDGE_TYPES,
    ProtocolMessage,
    metadata_to_vector,
)

if TYPE_CHECKING:  # pragma: no cover
    import torch
    from torch_geometric.data import Data as PyGData


NODE_TYPES: Final[tuple[str, ...]] = ("agent", "tool", "capability", "session")

# Heuristic to bucket a node-name into a partition when not explicitly tagged.
_TOOL_PREFIXES: Final[tuple[str, ...]] = ("tool:", "svc:", "server:", "api:")
_CAP_PREFIXES: Final[tuple[str, ...]] = ("cap:", "resource:", "prompt:")
_SESS_PREFIXES: Final[tuple[str, ...]] = ("sess:", "session:")


def _infer_node_type(name: str, tau: str, role: str) -> str:
    n = name.lower()
    if n.startswith(_TOOL_PREFIXES):
        return "tool"
    if n.startswith(_CAP_PREFIXES):
        return "capability"
    if n.startswith(_SESS_PREFIXES):
        return "session"
    if tau == "M_tool" and role == "dst":
        return "tool"
    if tau == "M_cap":
        return "capability" if role == "dst" else "agent"
    if tau == "M_ctrl" and role == "dst":
        return "session"
    return "agent"


@dataclass
class _Edge:
    src: int
    dst: int
    tau: str
    t_m: float
    feat: "torch.Tensor"  # (d_e,)


class ProtocolGraph:
    """In-memory typed temporal multigraph populated message-by-message."""

    NODE_TYPES: Final[tuple[str, ...]] = NODE_TYPES
    EDGE_TYPES: Final[tuple[str, ...]] = EDGE_TYPES

    def __init__(
        self,
        d_p: int = 384,
        d_metadata: int = len(DEFAULT_METADATA_SCHEMA),
        metadata_schema: Optional[list[str]] = None,
    ) -> None:
        self.d_p = d_p
        self.d_metadata = d_metadata
        self.d_e = d_p + d_metadata
        self.metadata_schema = list(metadata_schema or DEFAULT_METADATA_SCHEMA)

        self._name_to_id: dict[str, int] = {}
        self.node_type: dict[int, str] = {}
        self.node_attr: dict[int, dict[str, Any]] = {}
        self._edges: list[_Edge] = []
        self._type_counts: dict[str, int] = defaultdict(int)

    # ------------------------------------------------------------------ nodes
    def _add_node(self, name: str, ntype: str, attrs: Optional[dict] = None) -> int:
        if name in self._name_to_id:
            nid = self._name_to_id[name]
            if attrs:
                self.node_attr[nid].update(attrs)
            return nid
        nid = len(self._name_to_id)
        self._name_to_id[name] = nid
        self.node_type[nid] = ntype
        self.node_attr[nid] = dict(attrs or {})
        self._type_counts[ntype] += 1
        return nid

    # ----------------------------------------------------------------- edges
    def add_message(self, msg: ProtocolMessage) -> None:
        """Insert a message: upserts ``src``/``dst`` nodes and appends an edge."""
        import torch  # local

        s_type = _infer_node_type(msg.src, msg.tau, "src")
        d_type = _infer_node_type(msg.dst, msg.tau, "dst")
        s_id = self._add_node(msg.src, s_type)
        d_id = self._add_node(msg.dst, d_type)

        if msg.payload_emb is None:
            payload_vec = torch.zeros(self.d_p, dtype=torch.float32)
        else:
            pe = msg.payload_emb
            payload_vec = pe.detach().float() if hasattr(pe, "detach") else torch.as_tensor(pe, dtype=torch.float32)
            if payload_vec.numel() != self.d_p:
                # Truncate or zero-pad to d_p.
                buf = torch.zeros(self.d_p, dtype=torch.float32)
                k = min(self.d_p, payload_vec.numel())
                buf[:k] = payload_vec.flatten()[:k]
                payload_vec = buf

        meta_vec = metadata_to_vector(msg.metadata, self.metadata_schema)
        feat = torch.cat([payload_vec, meta_vec], dim=0)
        self._edges.append(_Edge(s_id, d_id, msg.tau, float(msg.t_m), feat))

    def extend(self, messages: Iterable[ProtocolMessage]) -> None:
        for m in messages:
            self.add_message(m)

    # --------------------------------------------------------------- queries
    def __len__(self) -> int:
        return len(self._edges)

    @property
    def num_nodes(self) -> int:
        return len(self._name_to_id)

    @property
    def num_edges(self) -> int:
        return len(self._edges)

    def node_id(self, name: str) -> int:
        return self._name_to_id[name]

    # ------------------------------------------------------------- windowing
    def windows(self, width: float, stride: float) -> Generator["ProtocolGraph", None, None]:
        """Yield sliding-window sub-graphs over edge timestamps."""
        if not self._edges:
            return
        t0 = min(e.t_m for e in self._edges)
        t1 = max(e.t_m for e in self._edges)
        start = t0
        while start <= t1:
            end = start + width
            sub = ProtocolGraph(self.d_p, self.d_metadata, self.metadata_schema)
            # Re-add edges within window; we copy nodes lazily via add_message
            # paths so the sub-graph keeps a compact node id-space.
            id_map: dict[int, str] = {v: k for k, v in self._name_to_id.items()}
            import torch  # noqa: F401

            for e in self._edges:
                if start <= e.t_m < end:
                    s_name = id_map[e.src]
                    d_name = id_map[e.dst]
                    s_id = sub._add_node(s_name, self.node_type[e.src])
                    d_id = sub._add_node(d_name, self.node_type[e.dst])
                    sub._edges.append(_Edge(s_id, d_id, e.tau, e.t_m, e.feat))
            if sub.num_edges:
                yield sub
            start += stride

    # ----------------------------------------------------------- PyG export
    def to_pyg_data(
        self,
        window: Optional[tuple[float, float]] = None,
    ) -> Any:
        """Return a ``torch_geometric.data.Data`` or dict fallback.

        Output contract (consumed by the GNN / Mamba encoder):
        * ``edge_index``  - LongTensor ``(2, E)``
        * ``edge_attr``   - FloatTensor ``(E, d_e)`` where ``d_e = d_p + d_mu``
        * ``edge_time``   - FloatTensor ``(E,)`` unix seconds
        * ``edge_type``   - LongTensor ``(E,)`` indexing :data:`EDGE_TYPES`
        * ``x``           - FloatTensor ``(N, num_node_types)`` one-hot type
        * ``node_type``   - LongTensor ``(N,)`` indexing :data:`NODE_TYPES`
        """
        import torch

        if window is not None:
            t_lo, t_hi = window
            edges = [e for e in self._edges if t_lo <= e.t_m < t_hi]
        else:
            edges = self._edges

        if edges:
            edge_index = torch.tensor([[e.src for e in edges], [e.dst for e in edges]], dtype=torch.long)
            edge_attr = torch.stack([e.feat for e in edges], dim=0)
            edge_time = torch.tensor([e.t_m for e in edges], dtype=torch.float32)
            edge_type = torch.tensor([EDGE_TYPES.index(e.tau) for e in edges], dtype=torch.long)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, self.d_e), dtype=torch.float32)
            edge_time = torch.zeros((0,), dtype=torch.float32)
            edge_type = torch.zeros((0,), dtype=torch.long)

        n = self.num_nodes
        node_type_ids = torch.tensor(
            [NODE_TYPES.index(self.node_type.get(i, "agent")) for i in range(n)],
            dtype=torch.long,
        )
        x = torch.zeros((n, len(NODE_TYPES)), dtype=torch.float32)
        if n:
            x[torch.arange(n), node_type_ids] = 1.0

        payload = {
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "edge_time": edge_time,
            "edge_type": edge_type,
            "x": x,
            "node_type": node_type_ids,
            "num_nodes": n,
        }
        try:
            from torch_geometric.data import Data  # type: ignore[import-not-found]
        except ImportError:
            return payload
        return Data(**payload)


__all__ = ["ProtocolGraph", "NODE_TYPES", "EDGE_TYPES"]
