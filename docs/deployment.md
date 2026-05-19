# Deployment

MambaGuard is delivered to production as **detector #14** in the **RobustIDPS.ai** platform (<https://github.com/rogerpanel/robustidps.ai>, Zenodo DOI [10.5281/zenodo.19129512](https://doi.org/10.5281/zenodo.19129512)). This document describes the runtime topology, the container stack, and the operational SLOs.

---

## 1. Platform integration

The RobustIDPS.ai gateway routes every observed protocol message through a chain of 14 detectors; MambaGuard is the last (catch-all) and the only certified one. The interface is a thin gRPC service:

```proto
service MambaGuard {
  rpc Score (ScoreRequest) returns (ScoreReply);
  rpc Certify (CertifyRequest) returns (CertifyReply);
  rpc Health (HealthRequest) returns (HealthReply);
}

message ScoreRequest {
  bytes  graph_snapshot = 1;   // serialised PyG TemporalData delta
  Message msg           = 2;   // canonicalised m = (tau, s, d, p, mu, t_m)
}

message ScoreReply {
  uint32  class_id        = 1; // index in the 34-class taxonomy
  float   class_prob      = 2;
  float   certified_radius = 3; // 0.0 if not yet certified
  string  policy_action   = 4; // sampled defender action from Hedge
  uint64  inference_us    = 5;
}
```

The Python reference implementation is in `mambaguard.deploy.grpc_server`; a thin FastAPI wrapper (`mambaguard.deploy.api`) is provided for HTTP consumers.

---

## 2. Container stack

```
                              +----------------------+
   public TLS  ---------->    |  Cloudflare Tunnel    |
                              +----------+-----------+
                                         |
                                         v
                              +----------------------+
                              |     FastAPI / gRPC    |    (mambaguard:1.0.0)
                              |  selective-scan kernel|
                              |   fp16  +  PyTorch 2.3|
                              +----+------------+----+
                                   |            |
                          stream   |            |  feature/event store
                                   v            v
                       +-----------+-+      +---+---------+
                       |    Redis    |      | PostgreSQL  |
                       |   streams   |      |    14+      |
                       +-------------+      +-------------+
```

Each component is published as a Docker image; the full stack is brought up by `docker compose -f deploy/compose.yaml up -d` (composition lives in the `robustidps.ai` sibling repository). Key components:

| Service | Image | Purpose |
| --- | --- | --- |
| `mambaguard-api` | `mambaguard:1.0.0` | gRPC + FastAPI; selective-scan kernel, fp16; one GPU |
| `redis` | `redis:7-alpine` | Stream broker, Hedge weight cache |
| `postgres` | `postgres:16` | Event store, certificate log, audit |
| `cloudflared` | `cloudflare/cloudflared:latest` | Zero-trust ingress (no public IP exposure) |
| `prometheus` + `grafana` | upstream | Latency / throughput / certificate dashboards |

Secrets and TLS material live in the `robustidps.ai` repo's `deploy/secrets/`. The MambaGuard container needs only a single env var, `MAMBAGUARD_CKPT`, pointing to a `.pt` checkpoint (mounted read-only).

---

## 3. SLOs

| SLO | Target | Measured |
| --- | --- | --- |
| **Inference latency (p50)** | < 5 ms / msg | **3.7 ms** on A100-80GB, batch 1, fp16 |
| **Inference latency (p99)** | < 10 ms / msg | 8.2 ms |
| **Throughput** | > 1 Mmsg/s | **1.13 Mmsg/s** on A100-80GB, batch 4096 |
| **Certified-radius freshness** | < 30 s | Background MC sweep every 10 s |
| **Stackelberg policy refresh** | < 60 s | LP solve takes < 2 s; refreshed every 30 s |
| **Availability** | 99.9 % | Quarterly SLO; tunnel + replica health-check |

Throughput numbers measured with the canonical `tools/bench_throughput.py` script (lives in the production repo) on the IDS-PQC stream.

---

## 4. Hardware

The production deployment uses a single **NVIDIA A100 80GB** (SXM4) per replica. Smaller deployments can run on **L4 / L40S** (latency ~7 ms, throughput ~0.4 Mmsg/s) or on **RTX 4090** (latency ~5 ms, throughput ~0.6 Mmsg/s); fp16 is required to meet the < 10 ms p99 SLO on any of these.

CPU-only inference is supported (via `pip install mamba-ssm --no-binary=:all: --no-deps` + the reference Python scan) but is **not** recommended; expect ~80 ms / msg and < 20 kmsg/s.

---

## 5. Observability

Prometheus metrics exposed on `:9090/metrics`:

- `mambaguard_inference_seconds` (histogram, labels: `class`, `protocol`)
- `mambaguard_throughput_msgs_total` (counter)
- `mambaguard_certified_radius` (gauge, labels: `sigma`)
- `mambaguard_hedge_regret` (gauge)
- `mambaguard_stackelberg_value` (gauge)
- `mambaguard_lipschitz_constant` (gauge)

A Grafana dashboard JSON is shipped in the production repo under `deploy/grafana/mambaguard.json`.

---

## 6. Upgrade procedure

1. Build & tag image: `docker build -t mambaguard:<x.y.z> .`
2. Push to registry.
3. Roll out one replica at a time; the gateway shifts traffic with a 30 s drain.
4. Verify SLOs in Grafana; if Macro-F1 on the canary stream regresses by > 0.5 pp, the rollout aborts automatically.

The certificate log is append-only; old certificates remain valid for audit even after a model upgrade.
