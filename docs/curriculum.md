# GPU & AI Infrastructure — Engineer Upskill Program

**Audience:** AI deployment engineers, platform engineers, and field engineers building private GPU clouds (campus, edge, or on-prem datacenter).

**Goal:** After this program, you can **spec hardware**, **install and validate GPU stacks**, **design multi-GPU and multi-node layouts**, **deploy AI workloads** (VMs, containers, Kubernetes), and **debug** performance and connectivity issues without guessing.

**Estimated depth:** 4–8 weeks part-time, or 2 weeks intensive with labs.

**Related platform docs:** [PrivateCloud architecture](https://github.com/verventech/PrivateCloud/blob/main/docs/ARCHITECTURE.md) · [Network topology](https://github.com/verventech/PrivateCloud/blob/main/docs/knowledge/01-network-topology.md) · [Console connectivity](https://github.com/verventech/PrivateCloud/blob/main/docs/knowledge/05-console-connectivity.md) · [Guacamole integration](https://github.com/verventech/PrivateCloud/blob/main/docs/knowledge/06-guacamole-integration.md)

---

## How to use this document

| Section | Use |
|---------|-----|
| **Modules 1–20** | Self-study and instructor-led theory |
| **[Appendix A: Lab Worksheets](#appendix-a-printable-lab-worksheets)** | Print one worksheet per lab; trainee fills by hand or in PDF |
| **[Appendix B: Answer keys & hints](#appendix-b-instructor-answer-keys--hints)** | Instructor reference only |

**Printing worksheets:** Open this file in a Markdown viewer or GitHub, jump to [Appendix A](#appendix-a-printable-lab-worksheets), and print from **Lab A1** through **Lab G1**. Each worksheet is separated by a page break (`---`).

---

## Table of contents

1. [What an AI deployment engineer owns](#1-what-an-ai-deployment-engineer-owns)
2. [GPU architecture fundamentals](#2-gpu-architecture-fundamentals)
3. [NVIDIA, AMD, and Intel for AI](#3-nvidia-amd-and-intel-for-ai)
4. [Motherboard, CPU, and platform dependencies](#4-motherboard-cpu-and-platform-dependencies)
5. [Power, cooling, and physical datacenter constraints](#5-power-cooling-and-physical-datacenter-constraints)
6. [The software stack (bottom to top)](#6-the-software-stack-bottom-to-top)
7. [Single-GPU deployment and validation](#7-single-gpu-deployment-and-validation)
8. [Multi-GPU on one host](#8-multi-gpu-on-one-host-clustering-at-the-node-level)
9. [Multi-node GPU clustering](#9-multi-node-gpu-clustering)
10. [Virtualization: VMs, passthrough, MIG, vGPU](#10-virtualization-vms-passthrough-mig-vgpu)
11. [Containers and Kubernetes GPU scheduling](#11-containers-and-kubernetes-gpu-scheduling)
12. [Training vs inference engineering](#12-training-vs-inference-engineering)
13. [Storage and data pipelines](#13-storage-and-data-pipelines)
14. [Networking for AI workloads](#14-networking-for-ai-workloads)
15. [Observability, benchmarking, and capacity planning](#15-observability-benchmarking-and-capacity-planning)
16. [Security, tenancy, and operations](#16-security-tenancy-and-operations)
17. [Hands-on lab curriculum](#17-hands-on-lab-curriculum)
18. [Troubleshooting playbook](#18-troubleshooting-playbook)
19. [Reference cheat sheets](#19-reference-cheat-sheets)
20. [Certification checklist](#20-certification-checklist)
- [Appendix A: Printable lab worksheets](#appendix-a-printable-lab-worksheets)
- [Appendix B: Instructor answer keys & hints](#appendix-b-instructor-answer-keys--hints)

---

## 1. What an AI deployment engineer owns

You sit between **hardware**, **platform**, and **ML teams**. You are not training models full-time, but you must understand what models need.

| Layer | You must be able to… |
|--------|----------------------|
| **Hardware** | Spec motherboards, GPUs, PSUs, cooling; validate PCIe topology |
| **OS / drivers** | Install NVIDIA/AMD drivers, CUDA/ROCm, kernel modules, persistence mode |
| **Virtualization** | GPU passthrough, MIG, vGPU, Proxmox/KVM patterns |
| **Containers** | NVIDIA Container Toolkit, device plugins, GPU memory limits |
| **Orchestration** | K8s scheduling, node labels, quotas, multi-tenant isolation |
| **Distributed AI** | NCCL, collective communication, InfiniBand vs RoCE |
| **Inference** | Ollama, vLLM, Triton, TensorRT, batching, quantization |
| **Operations** | Monitoring (DCGM, Prometheus), burn-in, driver upgrades, incident response |

**Core mindset:** GPUs are **throughput processors** with **tight coupling** to CPU PCIe lanes, memory bandwidth, network, and cooling. A slow link anywhere caps the whole system.

---

## 2. GPU architecture fundamentals

### 2.1 CPU vs GPU

| | CPU | GPU |
|---|-----|-----|
| **Design goal** | Low latency, complex branching | Massive parallelism |
| **Cores** | Few powerful cores (8–128) | Thousands of simpler cores |
| **Memory** | Large RAM, cache hierarchy | High-bandwidth VRAM, smaller capacity |
| **Best for** | OS, databases, control plane | Matrix math, convolutions, attention |

GPUs excel at **same operation on many data elements** (SIMT — Single Instruction, Multiple Threads).

### 2.2 Key NVIDIA building blocks

| Term | Meaning |
|------|---------|
| **SM** (Streaming Multiprocessor) | Basic compute unit; runs warps |
| **CUDA Core** | FP32/INT ALU inside an SM |
| **Warp** | 32 threads executing in lockstep |
| **Tensor Core** | Hardware matrix multiply (FP16/BF16/INT8/FP8) — critical for AI |
| **VRAM** | GPU-local memory; capacity limits model size |
| **Memory bandwidth** | GB/s between VRAM and SMs; often the inference bottleneck |
| **L2 cache** | Shared on-die cache; helps reuse |

**Engineer takeaway:** For LLMs, care about **VRAM capacity**, **memory bandwidth**, and **Tensor Core throughput** — not raw CUDA core count alone.

### 2.3 Memory hierarchy

1. **Registers** — fastest, per-thread (tiny)
2. **Shared memory / L1** — per-SM (CUDA)
3. **L2 cache** — on-die
4. **VRAM (global memory)** — weights and activations
5. **Host RAM** — via PCIe (slow for compute; used for staging)

**Unified Memory** can oversubscribe VRAM by paging to host RAM — convenient but **kills performance** in production.

### 2.4 Precision and AI performance

| Precision | Bits | Typical use |
|-----------|------|-------------|
| FP32 | 32 | Legacy training, some ops |
| FP16 / BF16 | 16 | Training & inference (BF16 better range) |
| FP8 | 8 | Hopper-class training/inference |
| INT8 / INT4 | 8/4 | Quantized inference (GPTQ, AWQ, GGUF) |

### 2.5 PCIe and the CPU–GPU link

| Gen | x16 bandwidth (per direction, approx.) |
|-----|----------------------------------------|
| PCIe 3.0 | ~16 GB/s |
| PCIe 4.0 | ~32 GB/s |
| PCIe 5.0 | ~64 GB/s |

---

## 3. NVIDIA, AMD, and Intel for AI

### 3.1 NVIDIA (dominant for AI today)

- **Consumer / prosumer:** RTX 4090, 5090 — labs, inference, fine-tuning; limited datacenter features
- **Datacenter:** A100, H100, H200, L40S — ECC, NVLink, MIG, enterprise drivers
- **Software:** CUDA, cuDNN, NCCL, TensorRT, Triton Inference Server

### 3.2 AMD

- **Hardware:** MI300X, MI250X; consumer/pro Radeon
- **Software:** ROCm, hipBLAS, RCCL
- **Reality:** Validate your exact model stack before committing

### 3.3 Intel

- **Hardware:** Gaudi, Arc, Xeon AMX
- **Software:** oneAPI, IPEX

### 3.4 Unified GPU platform (GPUaaS thinking)

Abstract vendor GPUs behind one portal/API. Engineers still must know driver differences, device plugins per vendor, and feature gaps (MIG, vGPU, RDMA).

---

## 4. Motherboard, CPU, and platform dependencies

### 4.1 PCIe lane budget (critical)

The CPU provides a **finite number of PCIe lanes**. Each GPU typically wants **x16**; NVMe uses **x4** per drive; high-speed NICs use **x8–x16**.

**Rule:** Draw a topology diagram **before purchase**.

### 4.2 Motherboard selection criteria

| Check | Why it matters |
|-------|----------------|
| **Physical spacing** | Double-wide GPUs need 2.5–3 slot spacing |
| **PCIe slot generation** | Gen4 GPU in Gen3 slot = half bandwidth |
| **ECC RAM support** | Datacenter stability |
| **IPMI / BMC** | Remote power cycle when GPU driver hangs |
| **M.2 vs PCIe sharing** | Some boards disable slots when M.2 populated |
| **Above 4G decoding / ReBAR** | Large GPU memory mapping |
| **VT-d / AMD-Vi (IOMMU)** | GPU passthrough to VMs |

### 4.3 CPU and RAM pairing

| Workload | Guidance |
|----------|----------|
| **Inference** | Moderate cores; prioritize PCIe lanes and NIC |
| **Training (single node)** | High core count for dataloaders |
| **Virtualization host** | Many cores; ample RAM for vCPUs |
| **NUMA (dual socket)** | Pin GPUs and NICs to same NUMA node |

Plan **2–4× GPU VRAM** as host RAM minimum for training nodes. Institutional lab hosts (e.g. 128 GB RAM + 32 GB VRAM) work for inference-heavy workloads; training often needs 256 GB+.

### 4.4 BIOS settings checklist

- Enable **VT-x / AMD-V**
- Enable **VT-d / IOMMU**
- Enable **Above 4G Decoding** and **Resizable BAR**
- Disable **CSM** (use UEFI)
- Set **PCIe link speed** to maximum supported

---

## 5. Power, cooling, and physical datacenter constraints

### 5.1 Power (indicative TDP)

| GPU class | TDP (approx.) |
|-----------|---------------|
| RTX 5090 | ~575 W |
| RTX 4090 | ~450 W |
| H100 SXM | ~700 W |
| A100 PCIe | ~300 W |

**Formula:** `Total ≈ (GPU TDP × count) + CPU + storage + fans + 20% headroom`

### 5.2 Cooling and burn-in

- Log **temperature and power** under sustained load
- Run **24h GPU stress** before production handover
- Watch for **thermal throttling** in benchmarks

---

## 6. The software stack (bottom to top)

```
Applications: PyTorch, vLLM, Ollama, etc.
Frameworks:   CUDA / ROCm, cuDNN, NCCL
Containers:   nvidia-container-toolkit
Orchestration: K8s device plugin, Slurm
Hypervisor:   KVM/Proxmox + VFIO passthrough
OS:           Ubuntu 22.04/24.04 LTS, RHEL
Driver:       NVIDIA 550+ (match CUDA version)
Firmware:     GPU VBIOS, motherboard BIOS
```

### Essential commands

```bash
nvidia-smi
nvidia-smi topo -m
nvidia-smi -q | less
lspci | grep -i nvidia
lsmod | grep nvidia
sudo nvidia-smi -pm 1    # persistence mode
```

---

## 7. Single-GPU deployment and validation

### Install sequence (Linux)

1. Minimal OS install (avoid nouveau conflicts)
2. Blacklist `nouveau` if needed
3. Install matching **NVIDIA driver**
4. Reboot → verify `nvidia-smi`
5. Install **CUDA toolkit** OR use container-only workflow
6. Install **NVIDIA Container Toolkit** for Docker/K8s
7. Run validation workload

### Validation

```bash
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name())"
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

---

## 8. Multi-GPU on one host (clustering at the node level)

### Connection types

| Link | Use |
|------|-----|
| **NVLink** | High-bandwidth multi-GPU training (DGX/H100) |
| **PCIe P2P** | Consumer multi-GPU |
| **Through host** | Slowest; when P2P disabled |

```bash
nvidia-smi topo -m
```

- **NV#** — NVLink
- **PIX** — PCIe through switch (OK)
- **PHB** — through host bridge (worse)
- **SYS** — across CPU sockets (worst)

### Multi-GPU inference patterns

| Pattern | When |
|---------|------|
| **Tensor parallelism** | Model too large for one GPU |
| **Pipeline parallelism** | Very deep models |
| **Data parallelism** | Multiple replicas for throughput |

### MIG (A100/H100 class)

Hardware partitions one GPU into isolated instances with QoS. Not available on most GeForce cards.

---

## 9. Multi-node GPU clustering

```
Node 1 (4× GPU) ──┐
Node 2 (4× GPU) ──┼── InfiniBand / RoCE / 100GbE ── K8s / Slurm / shared storage
Node N (8× GPU) ──┘
```

| Interconnect | AI training fit |
|--------------|-----------------|
| **InfiniBand HDR** | Excellent (RDMA) |
| **RoCEv2** | Good on lossless Ethernet |
| **10 GbE** | Inference / small scale only |

**Cluster checklist:** identical driver versions, NTP, shared filesystem, `NCCL_DEBUG=INFO` on first multi-node test.

---

## 10. Virtualization: VMs, passthrough, MIG, vGPU

| Model | Isolation | Use case |
|-------|-----------|----------|
| **Full passthrough** | 1 VM = 1 GPU | Researcher VMs, Windows CUDA labs |
| **MIG** | Hardware partitions | Many tenants on A100/H100 |
| **vGPU (GRID)** | Licensed slices | Enterprise VDI + ML |
| **Time-slicing (K8s)** | Software share | Dev/test only |

### VFIO passthrough (Proxmox/KVM)

1. Enable IOMMU in BIOS and kernel
2. Identify IOMMU groups
3. Bind GPU to `vfio-pci`
4. Attach PCI device to VM
5. Install guest driver

See [Console connectivity](https://github.com/verventech/PrivateCloud/blob/main/docs/knowledge/05-console-connectivity.md) and [Guacamole integration](https://github.com/verventech/PrivateCloud/blob/main/docs/knowledge/06-guacamole-integration.md) for browser desktop access patterns used in PrivateCloud.

---

## 11. Containers and Kubernetes GPU scheduling

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

| Component | Role |
|-----------|------|
| **NVIDIA GPU Operator** | Driver, device plugin, DCGM exporter |
| **Device plugin** | Exposes `nvidia.com/gpu` |
| **Runtime class** | `nvidia` runtime for GPU pods |

```yaml
resources:
  limits:
    nvidia.com/gpu: 1
```

**Multi-tenant patterns:** namespace quotas, taints on GPU nodes, DCGM → Prometheus, portal-enforced user limits (PrivateCloud model).

---

## 12. Training vs inference engineering

### Training needs

Multi-GPU, high VRAM, fast interconnect, checkpoint storage, BF16/FP16 mixed precision, NCCL allreduce.

### Inference needs

Low latency or high throughput, quantization, continuous batching (vLLM), KV cache tuning, local LLM fallback (Ollama).

### Rough VRAM sizing (weights only)

| Model | FP16 | INT4 approx. |
|-------|------|--------------|
| 7B | ~14 GB | ~4–5 GB |
| 13B | ~26 GB | ~8 GB |
| 70B | ~140 GB | multi-GPU required |

Add headroom for KV cache and batch size at inference time.

---

## 13. Storage and data pipelines

| Tier | Use |
|------|-----|
| **Local NVMe** | Model cache, container images |
| **Shared NAS (NFS)** | Datasets, home dirs |
| **Parallel FS** | Large-scale training |
| **Object (MinIO/S3)** | Artifacts, backups |

**Bottleneck:** dataloader starvation if storage cannot feed GPUs — use local cache and multiple workers.

---

## 14. Networking for AI workloads

| VLAN (campus private cloud) | Purpose |
|-----------------------------|---------|
| Management | IPMI, Proxmox, K3s API |
| Workload | VM/container networks |
| Storage | NFS/iSCSI |
| User access | Portal, Guacamole ingress |

East-west GPU traffic (NCCL) and north-south user traffic must be planned separately. Missing L3 routes between K8s nodes and VM subnets break consoles and RDP — see [network topology](https://github.com/verventech/PrivateCloud/blob/main/docs/knowledge/01-network-topology.md).

---

## 15. Observability, benchmarking, and capacity planning

| Metric | Tool |
|--------|------|
| GPU util / memory | `nvidia-smi`, DCGM |
| Temperature / power | `nvidia-smi dmon` |
| NCCL issues | `NCCL_DEBUG`, training logs |
| Inference latency | Prometheus histograms |

**Benchmarks:** gpu-burn (stability), nccl-tests (multi-GPU), llama.cpp bench (tokens/sec).

Plan **20–30% headroom** for thermal and burst lab sessions.

---

## 16. Security, tenancy, and operations

- **Tenant isolation:** VMs, K8s namespaces, network policies
- **Secrets:** encrypted settings, no keys in images
- **Driver updates:** staging node first; drain GPU workloads
- **CI/CD:** signed images (Jenkins → registry → K3s)
- **Compliance:** local Ollama when data cannot leave campus

---

## 17. Hands-on lab curriculum

| Module | Week | Labs |
|--------|------|------|
| **A — Hardware literacy** | 1 | A1–A4 |
| **B — Driver & CUDA** | 1–2 | B1–B3 |
| **C — Containers & K8s** | 2–3 | C1–C3 |
| **D — Virtualization** | 3 | D1–D3 |
| **E — Multi-GPU & distributed** | 4 | E1–E3 |
| **F — Production operations** | 4–5 | F1–F3 |
| **G — Capstone** | 5–6 | G1 |

Worksheets for every lab are in [Appendix A](#appendix-a-printable-lab-worksheets).

---

## 18. Troubleshooting playbook

| Symptom | Check | Fix |
|---------|-------|-----|
| Low GPU util | CPU, dataloader, storage | More workers, local cache |
| OOM on GPU | Model size, batch, KV cache | Quantize, reduce context, multi-GPU |
| NCCL hang | Network, P2P | `NCCL_DEBUG=INFO`, IB/RoCE config |
| VM no GPU | IOMMU, VFIO | Fix IOMMU group, hide hypervisor |
| Xid errors | Power, thermals, driver | `dmesg`, reseat GPU, PSU upgrade |
| Docker no GPU | Container toolkit | `nvidia-ctk runtime configure` |
| K8s pod pending | Device plugin, taints | `kubectl describe node` |

---

## 19. Reference cheat sheets

### Pre-purchase worksheet

```
Institution: ___________
Users (peak): ___________
Largest model (params): ___________
Windows VMs needed? Y/N
Offline LLM required? Y/N

GPUs: model ___ count ___
CPU: ___ (lanes: ___)
RAM: ___ GB
Storage: NVMe ___ TB + NAS ___ TB
Network: ___ Gb
Power: ___ kW available
```

### Node acceptance sign-off

- [ ] `nvidia-smi` — all GPUs visible
- [ ] PCIe x16 @ correct generation
- [ ] `nvidia-smi topo -m` documented
- [ ] 24h burn-in clean
- [ ] Container GPU test pass
- [ ] K8s GPU pod test pass
- [ ] Monitoring alerts configured
- [ ] Runbook delivered

### Glossary

| Term | Definition |
|------|------------|
| **CUDA** | NVIDIA parallel computing platform |
| **NCCL** | Multi-GPU communication library |
| **RDMA** | Remote Direct Memory Access |
| **MIG** | Hardware GPU partitioning |
| **VFIO** | Linux device passthrough framework |
| **Tensor Core** | Matrix acceleration unit |
| **KV cache** | Stored attention state during LLM inference |
| **Quantization** | Lower-precision weights for speed/size |

---

## 20. Certification checklist

An engineer is **deployment-ready** when they can, without assistance:

1. Spec a 2-GPU server with correct motherboard, PSU, and lane budget
2. Explain `nvidia-smi topo -m` and impact on training
3. Install drivers and validate PyTorch + container GPU test
4. Configure GPU passthrough VM on Proxmox
5. Deploy GPU workload on Kubernetes with quotas
6. Run Ollama (or vLLM) with offline fallback
7. Debug a failed NCCL job using logs and network checks
8. Present monitoring dashboard and capacity recommendation to a customer

---

## Recommended reading

| Resource | Focus |
|----------|--------|
| [NVIDIA CUDA Training / DLI](https://www.nvidia.com/en-us/training/) | CUDA, TensorRT, multi-GPU |
| [NVIDIA GPU Operator docs](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/) | K8s GPU production |
| [Proxmox PCI passthrough wiki](https://pve.proxmox.com/wiki/PCI_Passthrough) | VM GPU labs |
| [PyTorch distributed docs](https://pytorch.org/docs/stable/distributed.html) | DDP, FSDP |
| NCCL tests repository | Multi-node communication |

---

# Appendix A: Printable lab worksheets

> **Print instructions:** Print from this section. Each lab starts after a horizontal rule. Trainee name, date, and sign-off blocks are on every sheet. Photograph or scan completed worksheets for your training record.

---

## Lab A1 — PCIe lane map & platform design

**Module:** A — Hardware literacy  
**Duration:** 90 min  
**Objective:** Read a motherboard manual and produce a valid PCIe lane budget before hardware purchase.

| Field | Value |
|-------|-------|
| **Trainee name** | _________________________________ |
| **Date** | _________________________________ |
| **Instructor** | _________________________________ |
| **Hardware kit / SKU** | _________________________________ |

### Part 1 — Inventory

List every PCIe-consuming device planned for this build:

| # | Device | Slot / interface | Lanes required | PCIe gen |
|---|--------|------------------|----------------|----------|
| 1 | | | | |
| 2 | | | | |
| 3 | | | | |
| 4 | | | | |
| 5 | | | | |

**CPU model:** _________________________________  
**Total CPU PCIe lanes available:** _______  
**Chipset lane pool (if used):** _______

### Part 2 — Topology diagram

Sketch slots, CPUs, and devices (attach extra page if needed):

```
[ Draw here — label each link width: x16, x8, x4 and Gen3/4/5 ]





```

### Part 3 — Risk analysis

| Question | Your answer |
|----------|-------------|
| Will any GPU negotiate below x16? Why? | |
| Any M.2 slot that disables a PCIe slot? | |
| Is ReBAR / Above 4G required? Enabled how? | |
| Is IOMMU required for this design? | |

### Pass criteria (instructor)

- [ ] Lane budget balances (no impossible configuration)
- [ ] Diagram matches motherboard manual
- [ ] Risks documented with mitigation

**Trainee sign-off:** _________________________ **Date:** _________  
**Instructor sign-off:** _________________________ **Date:** _________

---

## Lab A2 — GPU physical install & link validation

**Module:** A — Hardware literacy  
**Duration:** 60 min  
**Objective:** Install GPU(s), verify PCIe detection and full link width.

| **Trainee name** | _________________________________ |
| **Date** | _________________________________ |
| **Host hostname / asset tag** | _________________________________ |

### Pre-install checklist

- [ ] PSU wattage documented: _______ W (GPUs: _______ W total)
- [ ] Power cables: native connectors used (no unsafe adapters)
- [ ] Slot spacing verified for GPU width
- [ ] ESD precautions followed

### Post-install commands (paste key output)

**`lspci \| grep -i nvidia`:**

```
(paste)

```

**`nvidia-smi -q` — PCIe link width & generation:**

| GPU index | Link width | Generation | GPU name |
|-----------|------------|------------|----------|
| 0 | | | |
| 1 | | | |

### Acceptance

- [ ] All GPUs visible in `nvidia-smi`
- [ ] Link width matches design (expected: _______)
- [ ] No errors in `dmesg` related to PCIe/GPU

**Trainee sign-off:** _________________________ **Instructor:** _________________________

---

## Lab A3 — BIOS platform configuration

**Module:** A — Hardware literacy  
**Duration:** 45 min  
**Objective:** Enable virtualization, IOMMU, and large BAR support for GPU workloads.

| **Trainee name** | _________________________________ |
| **Motherboard model** | _________________________________ |
| **BIOS version** | _________________________________ |

### Settings applied

| Setting | Required value | Actual value | ✓ |
|---------|----------------|--------------|---|
| Boot mode | UEFI (CSM off) | | [ ] |
| VT-x / AMD-V | Enabled | | [ ] |
| VT-d / IOMMU | Enabled | | [ ] |
| Above 4G Decoding | Enabled | | [ ] |
| Resizable BAR | Enabled | | [ ] |
| PCIe link speed | Max (Gen___) | | [ ] |

### OS verification

**`dmesg | grep -i iommu` (paste one line):**

```
```

**`/proc/cmdline` IOMMU flags:**

```
```

### Pass criteria

- [ ] IOMMU active in kernel
- [ ] GPU still at full PCIe width after reboot

**Trainee sign-off:** _________________________ **Instructor:** _________________________

---

## Lab A4 — Power & thermal budget

**Module:** A — Hardware literacy  
**Duration:** 45 min  
**Objective:** Calculate facility and PSU requirements for a multi-GPU node.

| **Trainee name** | _________________________________ |
| **Build description** | _________________________________ |

### Component power table

| Component | Qty | TDP/Rating (W) | Subtotal (W) |
|-----------|-----|----------------|--------------|
| GPU | | | |
| CPU | | | |
| Storage | | | |
| Fans/misc | | | |
| **Subtotal** | | | |
| **+20% headroom** | | | |
| **Total draw** | | | |

**PSU installed:** _______ W × _______ units = _______ W capacity  
**Margin:** _______ %  
**Rack/circuit limit:** _______ kW — within limit? Y / N

### Thermal (15 min load)

| GPU | Temp start | Temp peak | Power peak | Throttle? Y/N |
|-----|------------|-----------|------------|---------------|
| 0 | | | | |
| 1 | | | | |

**Trainee sign-off:** _________________________ **Instructor:** _________________________

---

## Lab B1 — NVIDIA driver install (clean host)

**Module:** B — Driver & CUDA  
**Duration:** 120 min  
**Objective:** Install driver on Ubuntu/RHEL; achieve stable `nvidia-smi`.

| **Trainee name** | _________________________________ |
| **OS version** | _________________________________ |

### Install log

| Step | Command / action | Result (pass/fail) |
|------|------------------|-------------------|
| Remove/blacklist nouveau | | |
| Install driver version | | |
| Reboot | | |
| `nvidia-smi` | | |
| Persistence mode `nvidia-smi -pm 1` | | |

**Driver version:** _________________________________  
**Kernel version:** _________________________________

### Issues encountered

| Issue | Resolution |
|-------|------------|
| | |

**Trainee sign-off:** _________________________ **Instructor:** _________________________

---

## Lab B2 — PyTorch GPU validation

**Module:** B — Driver & CUDA  
**Duration:** 60 min  
**Objective:** Run a minimal training loop with measurable GPU utilization.

| **Trainee name** | _________________________________ |

### Results

```bash
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

**Output:**

```
(paste)
```

| Metric | Value |
|--------|-------|
| Peak GPU util % | |
| Peak GPU memory (MiB) | |
| Training step time (ms) | |

- [ ] GPU util exceeded 70% during training step

**Trainee sign-off:** _________________________ **Instructor:** _________________________

---

## Lab B3 — Topology matrix interpretation

**Module:** B — Driver & CUDA  
**Duration:** 45 min  
**Objective:** Run and explain `nvidia-smi topo -m`.

| **Trainee name** | _________________________________ |
| **GPU count** | _______ |

**Paste topology matrix:**

```
(paste nvidia-smi topo -m)
```

### Written explanation (3–5 sentences)

Explain the path between GPU0 and GPU1 (or N/A for single GPU):

_______________________________________________________________________________

_______________________________________________________________________________

_______________________________________________________________________________

| Path type | Impact on NCCL / multi-GPU training |
|-----------|-----------------------------------|
| NVLink | |
| PIX | |
| PHB | |
| SYS | |

**Trainee sign-off:** _________________________ **Instructor:** _________________________

---

## Lab C1 — Docker GPU container

**Module:** C — Containers & K8s  
**Duration:** 45 min  
**Objective:** Run CUDA container with `--gpus all`.

| **Trainee name** | _________________________________ |

### Commands

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

**Output summary (GPU count, driver version):**

```
(paste)
```

- [ ] Container sees same GPU count as host
- [ ] `nvidia-container-toolkit` configured

**Trainee sign-off:** _________________________ **Instructor:** _________________________

---

## Lab C2 — Kubernetes GPU pod on K3s

**Module:** C — Containers & K8s  
**Duration:** 90 min  
**Objective:** Schedule a pod requesting `nvidia.com/gpu: 1`.

| **Trainee name** | _________________________________ |
| **Cluster** | _________________________________ |

### Deployment

**Namespace:** _________________________________  
**Node with GPU label:** _________________________________

```yaml
# Paste pod spec fragment (resources.limits)
```

| Check | Result |
|-------|--------|
| `kubectl describe node` shows allocatable GPU | |
| Pod status Running | |
| `nvidia-smi` inside pod works | |

**Trainee sign-off:** _________________________ **Instructor:** _________________________

---

## Lab C3 — Ollama local inference on GPU worker

**Module:** C — Containers & K8s  
**Duration:** 60 min  
**Objective:** Serve a local LLM on GPU; verify offline inference (no external API).

| **Trainee name** | _________________________________ |
| **Model pulled** | _________________________________ |

### Test procedure

1. Disconnect or block external API (describe method): _________________________________
2. Run inference request
3. Record latency and GPU memory

| Metric | Value |
|--------|-------|
| First token latency | |
| Tokens/sec (approx.) | |
| GPU memory used | |

- [ ] Inference succeeded without external API
- [ ] Documented fallback policy for institution

**Trainee sign-off:** _________________________ **Instructor:** _________________________

---

## Lab D1 — Proxmox GPU passthrough (Linux guest)

**Module:** D — Virtualization  
**Duration:** 120 min  
**Objective:** Pass GPU to Linux VM; run `nvidia-smi` in guest.

| **Trainee name** | _________________________________ |
| **Proxmox host** | _________________________________ |
| **VM ID** | _______ |

### IOMMU group

**GPU PCI address:** _________________________________  
**IOMMU group #:** _______  
**Other devices in group:** _________________________________

### VFIO binding

- [ ] Host driver unbound from GPU
- [ ] `vfio-pci` bound
- [ ] VM starts with GPU attached

**Guest `nvidia-smi` output (summary):**

```
(paste)
```

**Trainee sign-off:** _________________________ **Instructor:** _________________________

---

## Lab D2 — Windows VM + browser desktop (Guacamole)

**Module:** D — Virtualization  
**Duration:** 90 min  
**Objective:** Windows 11 GPU lab VM accessible via Guacamole/RDP through platform ingress.

| **Trainee name** | _________________________________ |
| **VM IP** | _________________________________ |

| Check | Pass? |
|-------|-------|
| RDP port reachable from guacd path | [ ] |
| Auto-login credentials configured | [ ] |
| Desktop visible in browser | [ ] |
| CUDA or GPU visible in guest (if applicable) | [ ] |

**Network path documented (K8s node → VM subnet):**

_______________________________________________________________________________

**Trainee sign-off:** _________________________ **Instructor:** _________________________

---

## Lab D3 — IOMMU conflict resolution

**Module:** D — Virtualization  
**Duration:** 60 min  
**Objective:** Document and fix an IOMMU group conflict.

| **Trainee name** | _________________________________ |

### Problem statement

```
(describe conflict — e.g. GPU grouped with USB controller)
```

### Resolution

| Action taken | Result |
|--------------|--------|
| | |
| | |

- [ ] Passthrough works after fix OR documented hardware limitation

**Trainee sign-off:** _________________________ **Instructor:** _________________________

---

## Lab E1 — Single-node multi-GPU PyTorch

**Module:** E — Multi-GPU & distributed  
**Duration:** 90 min  
**Objective:** Run `torchrun` across N GPUs; record scaling.

| **Trainee name** | _________________________________ |
| **GPU count N** | _______ |

| Config | Throughput (samples/sec) | Notes |
|--------|--------------------------|-------|
| 1 GPU | | |
| N GPUs | | |
| Speedup (N × single / multi) | | |

- [ ] Speedup is reasonable for topology (see B3)

**Trainee sign-off:** _________________________ **Instructor:** _________________________

---

## Lab E2 — NCCL all_reduce benchmark

**Module:** E — Multi-GPU & distributed  
**Duration:** 60 min  
**Objective:** Run `nccl-tests`; record bus bandwidth.

| **Trainee name** | _________________________________ |

```
(paste nccl-tests summary line)
```

| GPUs | Algorithm | Bus bandwidth (GB/s) |
|------|-----------|----------------------|
| | all_reduce | |

**Trainee sign-off:** _________________________ **Instructor:** _________________________

---

## Lab E3 — Multi-node job (optional)

**Module:** E — Multi-GPU & distributed  
**Duration:** 120 min  
**Objective:** Run a 2+ node distributed job OR document why hardware unavailable.

| **Trainee name** | _________________________________ |
| **Nodes** | _________________________________ |

| Check | Result |
|-------|--------|
| NCCL uses IB/RoCE (or Ethernet documented) | |
| Job completes without hang | |
| `NCCL_DEBUG=INFO` reviewed | |

**If skipped — reason:** _________________________________

**Trainee sign-off:** _________________________ **Instructor:** _________________________

---

## Lab F1 — GPU monitoring dashboard

**Module:** F — Production operations  
**Duration:** 90 min  
**Objective:** Expose GPU metrics to Prometheus/Grafana (or DCGM exporter).

| **Trainee name** | _________________________________ |

| Metric | Visible in dashboard? |
|--------|----------------------|
| GPU util % | [ ] |
| GPU memory | [ ] |
| Temperature | [ ] |
| Power | [ ] |

**Dashboard URL / screenshot ref:** _________________________________

**Trainee sign-off:** _________________________ **Instructor:** _________________________

---

## Lab F2 — 24-hour burn-in report

**Module:** F — Production operations  
**Duration:** 24h (wall clock)  
**Objective:** Stress GPUs; produce acceptance report.

| **Trainee name** | _________________________________ |
| **Host asset tag** | _________________________________ |
| **Start time** | _________________________________ |
| **End time** | _________________________________ |

| GPU | Errors (Xid) | Max temp °C | Throttle events |
|-----|--------------|-------------|-----------------|
| 0 | | | |
| 1 | | | |

**Recommendation:** Accept / Accept with notes / Reject  

**Notes:** _______________________________________________________________________________

**Trainee sign-off:** _________________________ **Instructor:** _________________________

---

## Lab F3 — Driver upgrade & rollback drill

**Module:** F — Production operations  
**Duration:** 90 min  
**Objective:** Document safe driver upgrade procedure for production GPU nodes.

| **Trainee name** | _________________________________ |

| Step | Action | Verified |
|------|--------|----------|
| 1 | Drain GPU workloads | [ ] |
| 2 | Snapshot / backup state | [ ] |
| 3 | Upgrade driver to version _______ | [ ] |
| 4 | Reboot + `nvidia-smi` | [ ] |
| 5 | Run smoke test (B2 or C1) | [ ] |
| 6 | Rollback procedure documented | [ ] |

**Rollback command summary:**

```
(paste)
```

**Trainee sign-off:** _________________________ **Instructor:** _________________________

---

## Lab G1 — Institutional capstone sign-off

**Module:** G — Capstone delivery  
**Duration:** 1–2 weeks  
**Objective:** Deliver a mini private AI cloud suitable for a university lab.

| **Trainee name** | _________________________________ |
| **Institution / scenario** | _________________________________ |
| **Capstone date** | _________________________________ |

### Deliverables checklist

| # | Deliverable | Done | Evidence (URL / doc / screenshot) |
|---|-------------|------|-----------------------------------|
| 1 | Hardware spec + lane diagram (A1) | [ ] | |
| 2 | GPU node burn-in passed (F2) | [ ] | |
| 3 | Proxmox + K3s operational | [ ] | |
| 4 | Portal/API self-service VM provision | [ ] | |
| 5 | GPU VM or GPU container for student workload | [ ] | |
| 6 | Ollama offline fallback demonstrated (C3) | [ ] | |
| 7 | Admin quotas + audit trail | [ ] | |
| 8 | Guacamole or console lab access (D2) | [ ] | |
| 9 | Monitoring dashboard (F1) | [ ] | |
| 10 | 5-page handover runbook | [ ] | |

### Oral defense (instructor notes)

| Question | Satisfactory? |
|----------|---------------|
| Explain PCIe lane bottleneck for this build | [ ] |
| Explain when to use VM vs container GPU | [ ] |
| Walk through incident: GPU OOM in student lab | [ ] |
| Explain Ollama vs external API tradeoffs | [ ] |

### Final grade

- [ ] **Certified** — ready for field deployment
- [ ] **Conditional** — remediate: _________________________________
- [ ] **Not yet** — repeat modules: _________________________________

**Trainee sign-off:** _________________________ **Date:** _________  
**Lead instructor sign-off:** _________________________ **Date:** _________

---

# Appendix B: Instructor answer keys & hints

> Do not distribute to trainees before lab completion.

| Lab | Key points |
|-----|------------|
| **A1** | Lane math must include chipset bifurcation; flag if second GPU runs x8 |
| **A2** | `LnkSta` in `nvidia-smi -q` must match design; downgraded links often BIOS or slot sharing |
| **A3** | Intel: `intel_iommu=on`; AMD: `amd_iommu=on`; verify DMAR/AMD-Vi in dmesg |
| **A4** | 5090-class builds often need 1200W+ PSU for dual GPU; reject <15% PSU margin |
| **B1** | `NVML: Driver/library version mismatch` → reboot or purge mismatched user-space libs |
| **B2** | Low util usually batch size too small or CPU-bound dataloader |
| **B3** | SYS between sockets on dual-CPU boards — expect poor NCCL without NVLink |
| **C1** | `nvidia-ctk runtime configure --runtime=docker` then restart docker |
| **C2** | Pending pods: check `nvidia-device-plugin` logs and node taints |
| **C3** | `ollama run llama3.2` then `ollama ps` shows GPU layer offloading |
| **D1** | Use `qm set <vmid> -hostpci0 <addr>,pcie=1` on Proxmox |
| **D2** | guacd often needs hostNetwork or route to VM subnet — see PrivateCloud knowledge/05, 06 |
| **D3** | ACS override is last resort; prefer motherboard with clean groups |
| **E1** | Ideal speedup rarely linear; <50% of theoretical on PCIe-only consumer GPUs is common |
| **E2** | Record topology alongside bandwidth for grading |
| **F2** | Any Xid 79/13 → power or thermal; document RMA path |
| **G1** | Capstone must include written runbook: driver version, contacts, backup, upgrade window |

---

*Document version: 1.0 · Last updated: 2026-09-01 · Repo: [verventech/gpu-ai-deployment-engineer](https://github.com/verventech/gpu-ai-deployment-engineer)*
