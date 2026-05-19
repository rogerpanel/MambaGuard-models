"""Dataset loaders and PyTorch wrappers for MambaGuard (Section 5).

Each loader returns an iterator of :class:`ProtocolMessage` records. URLs and
citations are listed inline per-loader; the loaders themselves never download
at import time. Pass ``root`` to a previously-extracted dataset directory.

Dataset URLs / citations
------------------------
* CSE-CIC-IDS2018 -- https://www.unb.ca/cic/datasets/ids-2018.html
  Sharafaldin et al., ICISSP 2018.
* CIC-IoT2023 -- https://www.unb.ca/cic/datasets/iotdataset-2023.html
  Neto et al., Sensors 2023.
* UNSW-NB15 -- https://research.unsw.edu.au/projects/unsw-nb15-dataset
  Moustafa & Slay, MilCIS 2015.
* Edge-IIoTset -- https://ieee-dataport.org/documents/edge-iiotset
  Ferrag et al., IEEE Access 2022.
* CICIDS2017 -- https://www.unb.ca/cic/datasets/ids-2017.html
  Sharafaldin et al., ICISSP 2018.
* NSL-KDD -- https://www.unb.ca/cic/datasets/nsl.html
  Tavallaee et al., CISDA 2009.
* CIC-DDoS2019 -- https://www.unb.ca/cic/datasets/ddos-2019.html
  Sharafaldin et al., ICCST 2019.
* NF-CSE-CIC-IDS2018-v3 PQC (IDS-PQC) -- Sarhan et al., 2024 (NetFlow v3
  re-encoding with post-quantum-TLS handshake captures appended).
* Kubernetes / Docker audit logs (component of ICS3D composite).
* Microsoft Cloud telemetry (component of ICS3D composite).
* AgentDojo -- https://github.com/ethz-spylab/agentdojo  (629 cases)
  Debenedetti et al., NeurIPS D&B 2024.
* InjecAgent -- https://github.com/uiuc-kang-lab/InjecAgent  (1054 cases)
  Zhan et al., ACL Findings 2024.
* ProtocolBench (this paper, MCP/ACP/A2A/cross-protocol JSONL scenarios).
* RobustIDPS-PQC -- Kaggle DOI 10.34740/kaggle/dsv/15424420.
* TGB 2.0 -- https://tgb.complexdatalab.com/  Huang et al., NeurIPS D&B 2024.

Composite datasets
------------------
* **IIS3D** = CSE-CIC-IDS2018 + CIC-IoT2023 + UNSW-NB15.
* **ICS3D** = Microsoft Cloud telemetry + Edge-IIoTset + K8s/Docker audit.
* **IDS-PQC** = NF-CSE-CIC-IDS2018-v3 PQC (single source, listed for symmetry).
"""
from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
)

from .canonicalize import (
    DEFAULT_METADATA_SCHEMA,
    ProtocolMessage,
    from_a2a,
    from_acp,
    from_anp,
    from_mcp,
    metadata_to_vector,
)
from .protocol_graph import EDGE_TYPES, ProtocolGraph
from .splits import temporal_split

if TYPE_CHECKING:  # pragma: no cover
    import torch
    from torch.utils.data import Dataset as _TorchDataset
else:
    try:  # pragma: no cover
        from torch.utils.data import Dataset as _TorchDataset
    except ImportError:  # pragma: no cover
        _TorchDataset = object  # type: ignore[assignment, misc]


_NOT_FOUND_HINT = (
    "Dataset root {root!r} not found. Download the dataset from the URL listed "
    "in the module docstring, extract it, and pass the extracted folder as `root`."
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_root(root: str) -> None:
    if not os.path.exists(root):
        raise FileNotFoundError(_NOT_FOUND_HINT.format(root=root))


def _csv_rows(path: str) -> Iterator[dict[str, str]]:
    with open(path, "r", newline="", encoding="utf-8", errors="replace") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            yield row


def _jsonl_rows(path: str) -> Iterator[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _find_files(root: str, extensions: Sequence[str]) -> list[str]:
    out: list[str] = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if fn.lower().endswith(tuple(extensions)):
                out.append(os.path.join(dp, fn))
    return sorted(out)


def _flow_to_msg(row: dict[str, str], tau: str, default_label_key: str = "Label") -> ProtocolMessage:
    """Map a CIC-style flow CSV row to a network-protocol message."""
    src = str(row.get("Source IP") or row.get("Src IP") or row.get("src_ip") or "src")
    dst = str(row.get("Destination IP") or row.get("Dst IP") or row.get("dst_ip") or "dst")
    proto = str(row.get("Protocol") or row.get("protocol") or "").lower()
    proto_name = {"6": "tcp", "17": "udp", "1": "icmp"}.get(proto, proto)
    meta = {
        "latency_ms": float(row.get("Flow Duration", 0) or 0) / 1000.0,
        "payload_bytes": float(row.get("Total Length of Fwd Packets", row.get("TotLen Fwd Pkts", 0)) or 0),
        "src_port": int(float(row.get("Source Port", row.get("Src Port", 0)) or 0)),
        "dst_port": int(float(row.get("Destination Port", row.get("Dst Port", 0)) or 0)),
        "protocol": proto_name,
        "rate_per_sec": float(row.get("Flow Packets/s", 0) or 0),
    }
    label_raw = row.get(default_label_key) or row.get("label") or row.get("Attack")
    label = 0 if str(label_raw).lower() in {"benign", "normal", "0"} else 1
    ts_raw = row.get("Timestamp") or row.get("timestamp") or "0"
    try:
        t_m = float(ts_raw)
    except ValueError:
        # Many CIC dumps use string timestamps; fall back to a stable hash.
        t_m = float(abs(hash(ts_raw)) % 10_000_000)
    return ProtocolMessage(
        tau=tau,
        src=src,
        dst=dst,
        payload=json.dumps({k: row.get(k) for k in list(row.keys())[:8]}),
        metadata=meta,
        t_m=t_m,
        label=label,
    )


# ---------------------------------------------------------------------------
# Network-flow loaders
# ---------------------------------------------------------------------------

def load_cse_cic_ids2018(root: str, split: str = "train") -> Iterator[ProtocolMessage]:
    """CSE-CIC-IDS2018 (https://www.unb.ca/cic/datasets/ids-2018.html)."""
    _require_root(root)
    msgs: list[ProtocolMessage] = []
    for path in _find_files(root, [".csv"]):
        for row in _csv_rows(path):
            msgs.append(_flow_to_msg(row, tau="M_data"))
    yield from _apply_split(msgs, split)


def load_cic_iot2023(root: str, split: str = "train") -> Iterator[ProtocolMessage]:
    """CIC-IoT2023 (https://www.unb.ca/cic/datasets/iotdataset-2023.html)."""
    _require_root(root)
    msgs = [
        _flow_to_msg(row, tau="M_data")
        for path in _find_files(root, [".csv"])
        for row in _csv_rows(path)
    ]
    yield from _apply_split(msgs, split)


def load_unsw_nb15(root: str, split: str = "train") -> Iterator[ProtocolMessage]:
    """UNSW-NB15 (https://research.unsw.edu.au/projects/unsw-nb15-dataset)."""
    _require_root(root)
    msgs = [
        _flow_to_msg(row, tau="M_data", default_label_key="label")
        for path in _find_files(root, [".csv"])
        for row in _csv_rows(path)
    ]
    yield from _apply_split(msgs, split)


def load_edge_iiotset(root: str, split: str = "train") -> Iterator[ProtocolMessage]:
    """Edge-IIoTset (https://ieee-dataport.org/documents/edge-iiotset)."""
    _require_root(root)
    msgs = [
        _flow_to_msg(row, tau="M_data", default_label_key="Attack_label")
        for path in _find_files(root, [".csv"])
        for row in _csv_rows(path)
    ]
    yield from _apply_split(msgs, split)


def load_cicids2017(root: str, split: str = "train") -> Iterator[ProtocolMessage]:
    """CICIDS2017 (https://www.unb.ca/cic/datasets/ids-2017.html)."""
    _require_root(root)
    msgs = [
        _flow_to_msg(row, tau="M_data")
        for path in _find_files(root, [".csv"])
        for row in _csv_rows(path)
    ]
    yield from _apply_split(msgs, split)


def load_nsl_kdd(root: str, split: str = "train") -> Iterator[ProtocolMessage]:
    """NSL-KDD (https://www.unb.ca/cic/datasets/nsl.html)."""
    _require_root(root)
    # NSL-KDD ships header-less .txt files; we synthesise minimal fields.
    msgs: list[ProtocolMessage] = []
    for path in _find_files(root, [".txt", ".csv"]):
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for i, line in enumerate(fh):
                cols = line.rstrip().split(",")
                if len(cols) < 5:
                    continue
                meta = {
                    "protocol": cols[1] if len(cols) > 1 else "",
                    "payload_bytes": float(cols[4]) if cols[4].replace(".", "", 1).isdigit() else 0.0,
                }
                label_raw = cols[-2] if len(cols) >= 2 else "normal"
                label = 0 if label_raw.lower() == "normal" else 1
                msgs.append(
                    ProtocolMessage(
                        tau="M_data",
                        src=f"nsl:{i}:src",
                        dst=f"nsl:{i}:dst",
                        payload=",".join(cols[:8]),
                        metadata=meta,
                        t_m=float(i),
                        label=label,
                    )
                )
    yield from _apply_split(msgs, split)


def load_cic_ddos2019(root: str, split: str = "train") -> Iterator[ProtocolMessage]:
    """CIC-DDoS2019 (https://www.unb.ca/cic/datasets/ddos-2019.html)."""
    _require_root(root)
    msgs = [
        _flow_to_msg(row, tau="M_data")
        for path in _find_files(root, [".csv"])
        for row in _csv_rows(path)
    ]
    yield from _apply_split(msgs, split)


def load_nf_cse_cic_ids2018_v3_pqc(root: str, split: str = "train") -> Iterator[ProtocolMessage]:
    """NF-CSE-CIC-IDS2018-v3 with post-quantum-TLS captures (IDS-PQC)."""
    _require_root(root)
    msgs: list[ProtocolMessage] = []
    for path in _find_files(root, [".csv"]):
        for row in _csv_rows(path):
            m = _flow_to_msg(row, tau="M_data")
            m.metadata["tls_pq"] = bool(row.get("TLS_PQ") or row.get("pq_kem"))
            msgs.append(m)
    yield from _apply_split(msgs, split)


# ---------------------------------------------------------------------------
# Audit / telemetry loaders
# ---------------------------------------------------------------------------

def load_kubernetes_audit(root: str, split: str = "train") -> Iterator[ProtocolMessage]:
    """Kubernetes / Docker audit JSON logs (ICS3D component)."""
    _require_root(root)
    msgs: list[ProtocolMessage] = []
    for path in _find_files(root, [".json", ".jsonl"]):
        rows = _jsonl_rows(path) if path.endswith(".jsonl") else iter([json.load(open(path))])
        for row in rows:
            user = (row.get("user") or {}).get("username", "system:unknown") if isinstance(row, dict) else "system"
            verb = row.get("verb", "get") if isinstance(row, dict) else "get"
            obj = row.get("objectRef", {}) if isinstance(row, dict) else {}
            dst = f"k8s:{obj.get('resource','?')}/{obj.get('name','?')}"
            tau = "M_ctrl" if verb in {"create", "delete", "patch", "update"} else "M_data"
            meta = {
                "auth": bool(row.get("annotations", {}).get("authorization.k8s.io/decision") == "allow"),
                "http_status": (row.get("responseStatus") or {}).get("code", 200),
            }
            ts = row.get("requestReceivedTimestamp") or row.get("stageTimestamp") or "0"
            try:
                t_m = float(ts) if isinstance(ts, (int, float)) else float(abs(hash(ts)) % 10_000_000)
            except (TypeError, ValueError):
                t_m = 0.0
            msgs.append(
                ProtocolMessage(
                    tau=tau,
                    src=str(user),
                    dst=dst,
                    payload=json.dumps(row)[:512],
                    metadata=meta,
                    t_m=t_m,
                    label=0 if row.get("responseStatus", {}).get("code", 200) < 400 else 1,
                )
            )
    yield from _apply_split(msgs, split)


def load_microsoft_cloud_telemetry(root: str, split: str = "train") -> Iterator[ProtocolMessage]:
    """Microsoft cloud telemetry / sign-in logs (ICS3D component)."""
    _require_root(root)
    msgs: list[ProtocolMessage] = []
    for path in _find_files(root, [".csv", ".jsonl"]):
        rows = _csv_rows(path) if path.endswith(".csv") else _jsonl_rows(path)
        for row in rows:
            label_raw = row.get("RiskLevel") or row.get("risk") or "low"
            label = 0 if str(label_raw).lower() in {"none", "low", "0"} else 1
            msgs.append(
                ProtocolMessage(
                    tau="M_comm",
                    src=str(row.get("UserPrincipalName", "user")),
                    dst=str(row.get("AppDisplayName", "app")),
                    payload=str(row.get("ConditionalAccessStatus", "")),
                    metadata={
                        "auth": str(row.get("Status", "")).lower() == "success",
                        "http_status": int(float(row.get("ResultType", 0) or 0)),
                    },
                    t_m=float(abs(hash(row.get("CreatedDateTime", ""))) % 10_000_000),
                    label=label,
                )
            )
    yield from _apply_split(msgs, split)


# ---------------------------------------------------------------------------
# LLM-agent benchmark loaders
# ---------------------------------------------------------------------------

def load_agentdojo(root: str, split: str = "train") -> Iterator[ProtocolMessage]:
    """AgentDojo benchmark (https://github.com/ethz-spylab/agentdojo)."""
    _require_root(root)
    msgs: list[ProtocolMessage] = []
    for path in _find_files(root, [".jsonl", ".json"]):
        rows = _jsonl_rows(path) if path.endswith(".jsonl") else [json.load(open(path))]
        for row in rows:
            for i, turn in enumerate(row.get("trace", [row])):
                rec = {
                    "method": f"tools/{turn.get('tool','call')}",
                    "params": {"client": turn.get("agent", "agent"), "server": turn.get("tool", "tool"),
                               "args": turn.get("args", {})},
                    "ts": float(i),
                    "label": 1 if row.get("injection_successful") else 0,
                    "id": f"{row.get('id','')}-{i}",
                }
                msgs.append(from_mcp(rec))
    yield from _apply_split(msgs, split)


def load_injecagent(root: str, split: str = "train") -> Iterator[ProtocolMessage]:
    """InjecAgent benchmark (https://github.com/uiuc-kang-lab/InjecAgent)."""
    _require_root(root)
    msgs: list[ProtocolMessage] = []
    for path in _find_files(root, [".jsonl", ".json"]):
        rows = _jsonl_rows(path) if path.endswith(".jsonl") else [json.load(open(path))]
        for row in rows:
            rec = {
                "method": "tools/invoke",
                "params": {"client": "user", "server": row.get("Tool", "tool"),
                           "args": row.get("User Instruction", row.get("instruction", ""))},
                "ts": 0.0,
                "label": 1 if row.get("Attack Type") else 0,
                "id": row.get("id"),
                "meta": {"auth": True},
            }
            msgs.append(from_mcp(rec))
    yield from _apply_split(msgs, split)


def load_protocolbench(root: str, split: str = "train") -> Iterator[ProtocolMessage]:
    """ProtocolBench (this paper). Local JSONL scenarios.

    Each line is ``{"protocol": "mcp"|"acp"|"a2a"|"anp", "record": {...}}``.
    Cross-protocol scenarios are represented as concatenated lines sharing a
    ``scenario_id`` field in ``record``.
    """
    _require_root(root)
    adapters: Dict[str, Callable[[dict], ProtocolMessage]] = {
        "mcp": from_mcp,
        "acp": from_acp,
        "a2a": from_a2a,
        "anp": from_anp,
    }
    msgs: list[ProtocolMessage] = []
    for path in _find_files(root, [".jsonl"]):
        for row in _jsonl_rows(path):
            proto = str(row.get("protocol", "mcp")).lower()
            adapter = adapters.get(proto, from_mcp)
            msgs.append(adapter(row.get("record", row)))
    yield from _apply_split(msgs, split)


def load_robustidps_pqc(root: str, split: str = "train") -> Iterator[ProtocolMessage]:
    """RobustIDPS-PQC (Kaggle DOI 10.34740/kaggle/dsv/15424420)."""
    _require_root(root)
    msgs: list[ProtocolMessage] = []
    for path in _find_files(root, [".csv"]):
        for row in _csv_rows(path):
            m = _flow_to_msg(row, tau="M_data")
            m.metadata["tls_pq"] = True
            msgs.append(m)
    yield from _apply_split(msgs, split)


def load_tgb2_temporal(root: str, split: str = "train") -> Iterator[ProtocolMessage]:
    """TGB 2.0 dynamic-graph benchmark (https://tgb.complexdatalab.com/)."""
    _require_root(root)
    msgs: list[ProtocolMessage] = []
    for path in _find_files(root, [".csv"]):
        for row in _csv_rows(path):
            try:
                t_m = float(row.get("ts", row.get("t", 0.0)) or 0.0)
            except ValueError:
                t_m = 0.0
            msgs.append(
                ProtocolMessage(
                    tau="M_comm",
                    src=str(row.get("u", row.get("src", "u"))),
                    dst=str(row.get("i", row.get("dst", "i"))),
                    payload=str(row.get("label", "")),
                    metadata={"payload_bytes": float(row.get("w", 0) or 0)},
                    t_m=t_m,
                    label=int(float(row.get("y", row.get("label", 0)) or 0) > 0),
                )
            )
    yield from _apply_split(msgs, split)


# ---------------------------------------------------------------------------
# Split application
# ---------------------------------------------------------------------------

def _apply_split(messages: list[ProtocolMessage], split: str) -> Iterator[ProtocolMessage]:
    if split == "all":
        yield from messages
        return
    sp = temporal_split(messages)
    if split not in sp:
        raise ValueError(f"unknown split={split!r}; expected one of {list(sp)} or 'all'")
    for i in sp[split]:
        yield messages[i]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

DATASET_REGISTRY: Dict[str, Callable[..., Iterator[ProtocolMessage]]] = {
    "cse_cic_ids2018": load_cse_cic_ids2018,
    "cic_iot2023": load_cic_iot2023,
    "unsw_nb15": load_unsw_nb15,
    "edge_iiotset": load_edge_iiotset,
    "cicids2017": load_cicids2017,
    "nsl_kdd": load_nsl_kdd,
    "cic_ddos2019": load_cic_ddos2019,
    "nf_cse_cic_ids2018_v3_pqc": load_nf_cse_cic_ids2018_v3_pqc,
    "kubernetes_audit": load_kubernetes_audit,
    "microsoft_cloud_telemetry": load_microsoft_cloud_telemetry,
    "agentdojo": load_agentdojo,
    "injecagent": load_injecagent,
    "protocolbench": load_protocolbench,
    "robustidps_pqc": load_robustidps_pqc,
    "tgb2_temporal": load_tgb2_temporal,
}


# ---------------------------------------------------------------------------
# Composite dataset
# ---------------------------------------------------------------------------

@dataclass
class CompositeDataset:
    """Concatenate several registered datasets (e.g. IIS3D, ICS3D, IDS-PQC).

    Loaders are called lazily on iteration; ``roots`` maps loader name to its
    on-disk root path. Splits are applied per-loader so temporal ordering is
    preserved within each source.
    """

    loaders: Sequence[str]
    roots: Dict[str, str]
    split: str = "train"

    def __iter__(self) -> Iterator[ProtocolMessage]:
        for name in self.loaders:
            if name not in DATASET_REGISTRY:
                raise KeyError(f"unknown loader {name!r}; choose from {list(DATASET_REGISTRY)}")
            if name not in self.roots:
                raise KeyError(f"no root configured for {name!r}")
            yield from DATASET_REGISTRY[name](self.roots[name], split=self.split)


# ---------------------------------------------------------------------------
# Torch Dataset + collate
# ---------------------------------------------------------------------------

class ProtocolMessageDataset(_TorchDataset):  # type: ignore[misc, valid-type]
    """``torch.utils.data.Dataset`` wrapper materialising ProtocolMessages.

    The dataset stores messages grouped into fixed-size *windows* of length
    ``window_size`` so that ``__getitem__`` returns a temporally-coherent
    sequence. Materialisation is eager (the iterator is drained at init).
    """

    def __init__(
        self,
        messages: Iterable[ProtocolMessage],
        window_size: int = 128,
        canonicaliser: Optional[Any] = None,
        compute_embeddings: bool = False,
    ) -> None:
        self.window_size = int(window_size)
        if self.window_size <= 0:
            raise ValueError("window_size must be > 0")
        self.messages: list[ProtocolMessage] = list(messages)
        if compute_embeddings and canonicaliser is not None:
            payloads = [m.payload for m in self.messages]
            emb = canonicaliser.encode_batch(payloads)
            for i, m in enumerate(self.messages):
                m.payload_emb = emb[i]
        self._n_windows = max(1, (len(self.messages) + self.window_size - 1) // self.window_size)

    def __len__(self) -> int:
        return self._n_windows

    def __getitem__(self, idx: int) -> list[ProtocolMessage]:
        if idx < 0 or idx >= self._n_windows:
            raise IndexError(idx)
        lo = idx * self.window_size
        hi = min(lo + self.window_size, len(self.messages))
        return self.messages[lo:hi]


def protocol_collate(batch: list[list[ProtocolMessage]]) -> dict[str, Any]:
    """Collate variable-length windows into padded tensors + a batched graph.

    Output dict keys (CONTRACT — training/model agents must honour these):

    * ``payload_emb``  -- FloatTensor ``(B, T_max, d_p)`` (zero where ``payload_emb`` missing)
    * ``metadata``     -- FloatTensor ``(B, T_max, d_metadata)``
    * ``edge_type``    -- LongTensor  ``(B, T_max)`` indexing :data:`EDGE_TYPES`
    * ``timestamps``   -- FloatTensor ``(B, T_max)``
    * ``attention_mask`` -- BoolTensor ``(B, T_max)``  True = valid token
    * ``labels``       -- LongTensor ``(B,)`` (-1 if window has no label;
                          window label = max over non-null message labels)
    * ``lengths``      -- LongTensor ``(B,)`` original window lengths
    * ``graph``        -- :class:`ProtocolGraph` built from the union of all
                          messages in the batch (use ``.to_pyg_data()`` to
                          tensorise for GNN consumption).
    """
    import torch

    B = len(batch)
    lengths = [len(w) for w in batch]
    T_max = max(lengths) if lengths else 0
    d_p = (batch[0][0].payload_emb.shape[-1]
           if batch and batch[0] and batch[0][0].payload_emb is not None
           else 384)
    d_mu = len(DEFAULT_METADATA_SCHEMA)

    payload_emb = torch.zeros((B, T_max, d_p), dtype=torch.float32)
    metadata = torch.zeros((B, T_max, d_mu), dtype=torch.float32)
    edge_type = torch.zeros((B, T_max), dtype=torch.long)
    timestamps = torch.zeros((B, T_max), dtype=torch.float32)
    attention_mask = torch.zeros((B, T_max), dtype=torch.bool)
    labels = torch.full((B,), -1, dtype=torch.long)

    graph = ProtocolGraph(d_p=d_p)

    for b, window in enumerate(batch):
        win_label = -1
        for t, m in enumerate(window):
            if m.payload_emb is not None:
                pe = m.payload_emb
                pe_t = pe.detach().float() if hasattr(pe, "detach") else torch.as_tensor(pe, dtype=torch.float32)
                k = min(d_p, pe_t.numel())
                payload_emb[b, t, :k] = pe_t.flatten()[:k]
            metadata[b, t] = metadata_to_vector(m.metadata)
            edge_type[b, t] = EDGE_TYPES.index(m.tau)
            timestamps[b, t] = float(m.t_m)
            attention_mask[b, t] = True
            if m.label is not None:
                win_label = max(win_label, int(m.label))
            graph.add_message(m)
        labels[b] = win_label

    return {
        "payload_emb": payload_emb,
        "metadata": metadata,
        "edge_type": edge_type,
        "timestamps": timestamps,
        "attention_mask": attention_mask,
        "labels": labels,
        "lengths": torch.tensor(lengths, dtype=torch.long),
        "graph": graph,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli() -> None:
    print("Available datasets (mambaguard.data.datasets):")
    for name in sorted(DATASET_REGISTRY):
        print(f"  - {name}")
    print("\nComposites:")
    print("  - iis3d  = cse_cic_ids2018 + cic_iot2023 + unsw_nb15")
    print("  - ics3d  = microsoft_cloud_telemetry + edge_iiotset + kubernetes_audit")
    print("  - ids_pqc = nf_cse_cic_ids2018_v3_pqc")


if __name__ == "__main__":  # pragma: no cover
    _cli()


__all__ = [
    "DATASET_REGISTRY",
    "CompositeDataset",
    "ProtocolMessageDataset",
    "protocol_collate",
    "load_cse_cic_ids2018",
    "load_cic_iot2023",
    "load_unsw_nb15",
    "load_edge_iiotset",
    "load_cicids2017",
    "load_nsl_kdd",
    "load_cic_ddos2019",
    "load_nf_cse_cic_ids2018_v3_pqc",
    "load_kubernetes_audit",
    "load_microsoft_cloud_telemetry",
    "load_agentdojo",
    "load_injecagent",
    "load_protocolbench",
    "load_robustidps_pqc",
    "load_tgb2_temporal",
]
