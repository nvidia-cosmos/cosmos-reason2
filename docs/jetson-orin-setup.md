# Cosmos-Reason2 on Jetson AGX Orin

This guide covers Jetson AGX Orin-specific setup for Cosmos-Reason2.

> **Prerequisites:** Complete the standard [Setup](../README.md#setup) steps (clone, uv, Hugging Face auth) first.

---

## Requirements

| Component | Version | Notes |
|-----------|---------|-------|
| Hardware | Jetson AGX Orin 64GB | |
| JetPack | 6.x | L4T R36.x |
| CUDA | 12.6 | Multiple versions can coexist |
| Python | 3.10 | Jetson wheels require 3.10 |
| cuDSS | 0.7.1+ | Not included in JetPack (see below) |

Verify your system:

```bash
cat /etc/nv_tegra_release          # Expected: R36, REVISION: 4.x
ls /usr/local/ | grep cuda-12.6    # Must exist
python3.10 --version               # Expected: 3.10.x
```

---

## Why a Separate Install?

Standard PyTorch wheels lack **sm_87** (Orin's compute architecture):

```
# PyPI torch: ['sm_80', 'sm_90', 'sm_100', 'sm_120']  ← missing sm_87
# Result: "CUDA error: no kernel image is available for execution on the device"
```

The `jp6` extra installs PyTorch, vLLM, and Triton from [Jetson AI Lab PyPI](https://pypi.jetson-ai-lab.io/) with sm_87 support.

---

## Installation

### 1. Install cuDSS (not included in JetPack)

```bash
wget https://developer.download.nvidia.com/compute/cudss/0.7.1/local_installers/cudss-local-repo-ubuntu2404-0.7.1_0.7.1-1_arm64.deb
sudo dpkg -i cudss-local-repo-ubuntu2404-0.7.1_0.7.1-1_arm64.deb
sudo cp /var/cudss-local-repo-ubuntu2404-0.7.1/cudss-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update && sudo apt-get -y install cudss
```

> For Ubuntu 22.04, replace `ubuntu2404` with `ubuntu2204`.

### 2. Configure CUDA Environment

Triton (required by vLLM) needs access to CUDA headers and `ptxas`. Add to `~/.bashrc` or `~/.zshrc`:

```bash
# CUDA environment for Triton/vLLM
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export CPATH=$CUDA_HOME/include:$CPATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:/usr/local/cuda-12.6/targets/aarch64-linux/lib:$LD_LIBRARY_PATH
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
```

Then reload your shell:

```bash
source ~/.zshrc  # or ~/.bashrc
```

### 3. Install with jp6 Extra

```bash
uv sync --extra jp6 --python 3.10
source .venv/bin/activate
```

### 4. Verify Installation

```bash
python3 -c "import torch; print(f'Arch: {torch.cuda.get_arch_list()}, Available: {torch.cuda.is_available()}')"
# Expected: Arch: ['sm_87'], Available: True

python3 -c "import vllm; print(f'vLLM: {vllm.__version__}')"
# Expected: vLLM: 0.14.0+cu126
```

---

## Running vLLM Server

### Quick Start

```bash
vllm serve nvidia/Cosmos-Reason2-2B \
  --allowed-local-media-path "$(pwd)" \
  --max-model-len 8192 \
  --media-io-kwargs '{"video": {"num_frames": -1}}' \
  --reasoning-parser qwen3 \
  --port 8000 \
  --gpu-memory-utilization 0.8
```

### Recommended Settings for AGX Orin

| Parameter | Value | Notes |
|-----------|-------|-------|
| `--max-model-len` | 8192 | Lower context saves memory; 16384 possible on 64GB |
| `--gpu-memory-utilization` | 0.8 | Leave headroom for system; adjust based on other processes |

### Testing the Server

```bash
# Check server is running
curl http://localhost:8000/v1/models

# Text-only query
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nvidia/Cosmos-Reason2-2B",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'

# Image query (using local file path)
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nvidia/Cosmos-Reason2-2B",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "image_url", "image_url": {"url": "file:///path/to/image.png"}},
        {"type": "text", "text": "What do you see in this image?"}
      ]
    }]
  }'
```

### Known Warnings (Safe to Ignore)

During startup you may see:

```
NvMapMemAllocInternalTagged: 1075072515 error 12
NvMapMemHandleAlloc: error 0
```

These are harmless CUDA graph memory allocation warnings. They occur because vLLM pre-allocates memory for various batch sizes and some allocations fail due to limited memory. The server handles this gracefully.

---

## Transformers Inference

For direct Transformers inference (without vLLM):

```bash
python scripts/inference_sample.py
```

---

## Feature Availability

| Feature | Orin | Notes |
|---------|------|-------|
| Transformers inference | ✅ | Full support |
| vLLM serving | ✅ | vLLM 0.14.0+cu126 from Jetson AI Lab |
| cosmos-rl post-training | ⚠️ | vLLM works; other deps may need testing |
| TRL post-training | ⚠️ | May require additional setup |
| Quantization | ⚠️ | Limited backend support |

---

## Troubleshooting

| Error | Cause | Solution |
|-------|-------|----------|
| `no kernel image is available` | PyTorch missing sm_87 | `rm -rf .venv && uv sync --extra jp6 --python 3.10` |
| `undefined symbol: __nvJitLinkGetErrorLogSize_12_9` | Loading wrong CUDA libs | Set `LD_LIBRARY_PATH` to CUDA 12.6 (see step 2) |
| `libcudss.so.0: cannot open` | Missing cuDSS | Install cuDSS (see step 1) |
| `No module named 'triton'` | Triton not installed | Verify `jp6` extra was used during install |
| `Cannot find ptxas` | CUDA bin not in PATH | Set `PATH=$CUDA_HOME/bin:$PATH` (see step 2) |
| `cuda.h: No such file or directory` | CUDA headers not found | Set `CPATH=$CUDA_HOME/include:$CPATH` (see step 2) |
| `Free memory less than desired` | Other processes using GPU | Lower `--gpu-memory-utilization` or close other apps |
| nvidia-smi vs nvcc version mismatch | Expected on Jetson | nvidia-smi shows driver max, nvcc shows toolkit |

---

## Orin vs Thor

| | Orin | Thor |
|-|------|------|
| Driver | iGPU (integrated) | SBSA (server-standard) |
| JetPack | 6.x | 7.x |
| CUDA | ≤12.9 | 13.0+ |
| PyPI Index | `jp6/*` | `sbsa/*` |
| vLLM | ✅ (0.14.0+) | ✅ |

---

## References

- [Jetson AI Lab PyPI](https://pypi.jetson-ai-lab.io/)
- [JetPack 6 Release Notes](https://docs.nvidia.com/jetson/jetpack/release-notes/)
