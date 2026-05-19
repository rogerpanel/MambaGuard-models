"""Protocol-message canonicalisation (Section 3.1 of the MambaGuard paper).

Each external protocol record (MCP, ACP, A2A, ANP, network flow, audit log) is
mapped to a uniform :class:`ProtocolMessage` carrying:

* ``tau``      - typed channel tag in ``{M_tool, M_comm, M_cap, M_data, M_ctrl}``
* ``src/dst``  - canonical agent / tool / capability identifiers
* ``payload``  - raw text content (encoded by a frozen sentence-transformer)
* ``metadata`` - numeric / categorical side-channel features
* ``t_m``      - unix timestamp (float seconds)

The payload is embedded with :class:`MessageCanonicaliser` (default
``sentence-transformers/all-MiniLM-L6-v2``, d_p=384). Metadata is mapped via
:func:`metadata_to_vector` against a fixed schema yielding a d_mu-dim vector;
edge features are the concatenation [payload_emb || metadata_vec].
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final, Optional, Union

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np
    import torch

EDGE_TYPES: Final[tuple[str, ...]] = ("M_tool", "M_comm", "M_cap", "M_data", "M_ctrl")

# HTTP status one-hot buckets used by the default metadata schema.
HTTP_STATUS_BUCKETS: Final[tuple[str, ...]] = (
    "1xx", "2xx", "3xx", "4xx", "5xx",
)

DEFAULT_METADATA_SCHEMA: Final[list[str]] = [
    "latency_ms",        # float
    "payload_bytes",     # float
    "auth_flag",         # bool -> {0,1}
    "tls_pq_flag",       # bool, post-quantum TLS in IDS-PQC
    "is_internal",       # bool
    "retry_count",       # int
    "rate_per_sec",      # float
    "src_port",          # int
    "dst_port",          # int
    "proto_tcp",         # one-hot transport (icmp falls into the all-zero bucket)
    "proto_udp",
    # http_status one-hot (5 dims) -> total = 11 + 5 = 16 -> d_mu = 16
    *(f"http_{b}" for b in HTTP_STATUS_BUCKETS),
]
assert len(DEFAULT_METADATA_SCHEMA) == 16, "d_mu must equal 16 by paper convention"


@dataclass
class ProtocolMessage:
    """A canonicalised inter-agent message (one directed edge in G(t))."""

    tau: str
    src: str
    dst: str
    payload: str
    metadata: dict[str, Any]
    t_m: float
    label: Optional[int] = None  # optional ground-truth class id
    payload_emb: Optional[Union["np.ndarray", "torch.Tensor"]] = None
    msg_id: Optional[str] = None
    extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.tau not in EDGE_TYPES:
            raise ValueError(
                f"tau={self.tau!r} not in {EDGE_TYPES}. Use the protocol adapters."
            )


# ---------------------------------------------------------------------------
# Payload embedding
# ---------------------------------------------------------------------------

class MessageCanonicaliser:
    """Frozen sentence-transformer encoder for raw text payloads.

    The underlying model is loaded lazily on the first :meth:`encode_batch`
    call so that simply importing this module is dependency-free.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
        d_p: int = 384,
        batch_size: int = 64,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.d_p = d_p
        self.batch_size = batch_size
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "sentence-transformers is required for MessageCanonicaliser; "
                "install with `pip install sentence-transformers`."
            ) from exc
        model = SentenceTransformer(self.model_name, device=self.device)
        for p in model.parameters():
            p.requires_grad = False
        model.eval()
        self._model = model
        return model

    def encode_batch(self, payloads: list[str]) -> "torch.Tensor":
        """Return a ``(N, d_p)`` float tensor of frozen payload embeddings."""
        import torch  # local import to keep module light

        if not payloads:
            return torch.zeros((0, self.d_p), dtype=torch.float32)
        model = self._load()
        emb = model.encode(
            payloads,
            batch_size=self.batch_size,
            convert_to_tensor=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return emb.float()


# ---------------------------------------------------------------------------
# Metadata vectorisation
# ---------------------------------------------------------------------------

def _http_status_bucket(status: Any) -> str:
    try:
        code = int(status)
    except (TypeError, ValueError):
        return ""
    if 100 <= code < 200:
        return "1xx"
    if 200 <= code < 300:
        return "2xx"
    if 300 <= code < 400:
        return "3xx"
    if 400 <= code < 500:
        return "4xx"
    if 500 <= code < 600:
        return "5xx"
    return ""


def metadata_to_vector(
    meta: dict[str, Any],
    schema: list[str] = DEFAULT_METADATA_SCHEMA,
) -> "torch.Tensor":
    """Map a metadata dict to a fixed-dim ``float32`` tensor following ``schema``.

    Missing keys default to ``0.0``; unknown protocol / status values are
    silently ignored. The schema is the authoritative ordering and must match
    the model's ``d_metadata`` config.
    """
    import torch  # local import

    bucket = _http_status_bucket(meta.get("http_status"))
    proto = str(meta.get("protocol", "")).lower()

    vec = [0.0] * len(schema)
    for i, key in enumerate(schema):
        if key == "latency_ms":
            vec[i] = float(meta.get("latency_ms", 0.0))
        elif key == "payload_bytes":
            vec[i] = float(meta.get("payload_bytes", 0.0))
        elif key == "auth_flag":
            vec[i] = 1.0 if meta.get("auth") else 0.0
        elif key == "tls_pq_flag":
            vec[i] = 1.0 if meta.get("tls_pq") else 0.0
        elif key == "is_internal":
            vec[i] = 1.0 if meta.get("is_internal") else 0.0
        elif key == "retry_count":
            vec[i] = float(meta.get("retry_count", 0))
        elif key == "rate_per_sec":
            vec[i] = float(meta.get("rate_per_sec", 0.0))
        elif key == "src_port":
            vec[i] = float(meta.get("src_port", 0))
        elif key == "dst_port":
            vec[i] = float(meta.get("dst_port", 0))
        elif key == "proto_tcp":
            vec[i] = 1.0 if proto == "tcp" else 0.0
        elif key == "proto_udp":
            vec[i] = 1.0 if proto == "udp" else 0.0
        elif key.startswith("http_"):
            vec[i] = 1.0 if key == f"http_{bucket}" else 0.0
    return torch.tensor(vec, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Protocol-specific adapters (skeletons - extend per dataset)
# ---------------------------------------------------------------------------

def from_mcp(record: dict[str, Any]) -> ProtocolMessage:
    """Map a Model Context Protocol JSON record.

    Field mapping
    -------------
    ``method``        -> ``tau`` (``tools/*`` -> ``M_tool``; ``resources/*`` /
                        ``prompts/*`` / ``initialize`` -> ``M_cap``).
    ``params.client`` -> ``src`` ; ``params.server`` -> ``dst``.
    ``params``        -> ``payload`` (JSON-serialised).
    ``meta``          -> ``metadata`` (latency, auth, ...).
    ``ts``            -> ``t_m`` (unix seconds).
    """
    method = str(record.get("method", ""))
    if method.startswith("tools/"):
        tau = "M_tool"
    elif method.startswith(("resources/", "prompts/", "initialize")):
        tau = "M_cap"
    else:
        tau = "M_ctrl"
    import json

    params = record.get("params", {}) or {}
    return ProtocolMessage(
        tau=tau,
        src=str(params.get("client", record.get("src", "client"))),
        dst=str(params.get("server", record.get("dst", "server"))),
        payload=json.dumps(params, default=str),
        metadata=dict(record.get("meta", {})),
        t_m=float(record.get("ts", record.get("timestamp", 0.0))),
        label=record.get("label"),
        msg_id=record.get("id"),
    )


def from_acp(record: dict[str, Any]) -> ProtocolMessage:
    """Agent Communication Protocol record -> ``M_comm`` message.

    Field mapping: ``sender->src``, ``receiver->dst``, ``content->payload``,
    ``timestamp->t_m``, side-channel fields under ``meta``.
    """
    return ProtocolMessage(
        tau="M_comm",
        src=str(record.get("sender", record.get("src", "agent_a"))),
        dst=str(record.get("receiver", record.get("dst", "agent_b"))),
        payload=str(record.get("content", record.get("payload", ""))),
        metadata=dict(record.get("meta", {})),
        t_m=float(record.get("timestamp", record.get("ts", 0.0))),
        label=record.get("label"),
        msg_id=record.get("id"),
    )


def from_a2a(record: dict[str, Any]) -> ProtocolMessage:
    """Google Agent-to-Agent record -> ``M_comm`` message.

    Field mapping: ``from->src``, ``to->dst``, ``message->payload``,
    ``time->t_m``.
    """
    return ProtocolMessage(
        tau="M_comm",
        src=str(record.get("from", record.get("src", "agent_a"))),
        dst=str(record.get("to", record.get("dst", "agent_b"))),
        payload=str(record.get("message", record.get("payload", ""))),
        metadata=dict(record.get("meta", {})),
        t_m=float(record.get("time", record.get("ts", 0.0))),
        label=record.get("label"),
        msg_id=record.get("id"),
    )


def from_anp(record: dict[str, Any]) -> ProtocolMessage:
    """Agent Network Protocol record.

    ``kind`` field selects channel:
    * ``"data"``   -> ``M_data``
    * anything else (``"control"``, ``"heartbeat"``) -> ``M_ctrl``
    """
    kind = str(record.get("kind", "control")).lower()
    tau = "M_data" if kind == "data" else "M_ctrl"
    return ProtocolMessage(
        tau=tau,
        src=str(record.get("src", "node_a")),
        dst=str(record.get("dst", "node_b")),
        payload=str(record.get("payload", "")),
        metadata=dict(record.get("meta", {})),
        t_m=float(record.get("ts", record.get("timestamp", 0.0))),
        label=record.get("label"),
        msg_id=record.get("id"),
    )


__all__ = [
    "EDGE_TYPES",
    "DEFAULT_METADATA_SCHEMA",
    "HTTP_STATUS_BUCKETS",
    "ProtocolMessage",
    "MessageCanonicaliser",
    "metadata_to_vector",
    "from_mcp",
    "from_acp",
    "from_a2a",
    "from_anp",
]
