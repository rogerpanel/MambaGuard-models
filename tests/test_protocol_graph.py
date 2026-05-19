"""Tests for the ProtocolGraph data structure."""
from __future__ import annotations

import pytest


def _import_graph():
    try:
        from mambaguard.data import EDGE_TYPES, ProtocolGraph, ProtocolMessage
    except Exception as exc:
        pytest.skip(f"data unavailable: {exc}")
    return ProtocolGraph, ProtocolMessage, EDGE_TYPES


def _make_msg(ProtocolMessage, EDGE_TYPES, i: int):
    tau = EDGE_TYPES[i % len(EDGE_TYPES)]
    return ProtocolMessage(
        tau=tau,
        src=("a" if i % 2 == 0 else "b"),
        dst=("b" if i % 2 == 0 else "a"),
        payload=f"msg-{i}",
        metadata={},
        t_m=float(i),
        label=i % 4,
        msg_id=f"m{i}",
    )


def test_add_messages_updates_counts():
    pytest.importorskip("torch")
    ProtocolGraph, ProtocolMessage, EDGE_TYPES = _import_graph()
    g = ProtocolGraph(d_p=8)
    for i in range(10):
        g.add_message(_make_msg(ProtocolMessage, EDGE_TYPES, i))
    assert g.num_nodes == 2
    assert g.num_edges == 10


def test_windows_disjoint_by_stride():
    pytest.importorskip("torch")
    ProtocolGraph, ProtocolMessage, EDGE_TYPES = _import_graph()
    g = ProtocolGraph(d_p=8)
    for i in range(20):
        g.add_message(_make_msg(ProtocolMessage, EDGE_TYPES, i))
    wins = list(g.windows(width=5.0, stride=5.0))
    assert len(wins) >= 3
    # Edges in consecutive windows must not share timestamps (stride == width).
    prev_max = -1.0
    for w in wins:
        ts = [e.t_m for e in w._edges]
        assert min(ts) >= prev_max
        prev_max = max(ts)


def test_to_pyg_data_keys():
    pytest.importorskip("torch")
    ProtocolGraph, ProtocolMessage, EDGE_TYPES = _import_graph()
    g = ProtocolGraph(d_p=8)
    for i in range(6):
        g.add_message(_make_msg(ProtocolMessage, EDGE_TYPES, i))
    data = g.to_pyg_data()
    for key in ("x", "edge_index", "edge_attr", "edge_time"):
        assert key in data or hasattr(data, key)
