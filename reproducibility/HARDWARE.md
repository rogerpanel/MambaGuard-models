# Hardware

## Reference platform (paper numbers)

| component | spec                                        |
|-----------|---------------------------------------------|
| GPU       | 1x NVIDIA A100-SXM4-80GB (80 GB HBM2e)      |
| CPU       | AMD EPYC 7763 64-core @ 2.45 GHz             |
| RAM       | 512 GB DDR4-3200 ECC                        |
| Disk      | 4 TB NVMe (Samsung PM9A3), XFS              |
| Network   | 100 GbE (only used for dataset download)    |

All latency / throughput numbers in the paper were collected on this
configuration with the GPU isolated (no other CUDA contexts active) and
SMT enabled on the host CPU.

## Minimum supported configuration

| component | minimum |
|-----------|---------|
| GPU       | 1x NVIDIA RTX 4090 (24 GB) or A6000 (48 GB) |
| CPU       | 16 physical cores                           |
| RAM       | 64 GB                                       |
| Disk      | 500 GB free for raw + processed datasets    |

### Scaling notes for smaller GPUs

- **24 GB cards (RTX 4090, A5000):** reduce `training.batch_size` from the
  default (256) to 64 and set `model.activation_checkpointing: true` in the
  config. Expect ~3.5x longer training wall-clock.
- **48 GB cards (A6000, L40S):** reduce `training.batch_size` to 128. Throughput
  scales roughly linearly with effective batch size.
- **Multi-GPU:** `MambaGuardTrainer` supports DDP via
  `torchrun --nproc-per-node=N -m scripts.train ...` Tested up to 8x A100.

## CPU-only fallback

The repository runs on CPU (no `mamba_ssm` GPU kernels) for the tiny test
configurations in `tests/`. End-to-end training on CPU is not recommended:
one epoch on IIS3D takes >24 h.
