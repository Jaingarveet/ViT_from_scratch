# ViT-from-scratch

## Motivation
This repository implements a Vision Transformer (ViT) entirely from scratch to build a strict mathematical and engineering foundation in low-level tensor manipulation. Rather than relying on high-level abstractions, this project explicitly defines the core operations of the architecture. It serves as a foundational stepping stone before scaling up to custom Vision-Language Models (VLMs) and multi-modal pipelines.

---

## Core Implementations

* **Architectural Components:** Built custom modules for Patch Extraction (via stride-based 2D convolutions), Class Token parameterization, Positional Embeddings, Transformer Encoders, and the final MLP head.
* **Optimized Multi-Head Attention:** Eliminated sequential loops in the MSA layer by projecting and reshaping queries, keys, and values into `(3, N, num_heads, num_patches, head_dim)` tensors, utilizing `.permute()` and `torch.matmul()` for highly parallelized batched operations.
* **Hardware Profiling:** Analyzed forward/backward pass bottlenecks using the PyTorch Profiler, interpreting Chrome traces, CUDA synchronization, and profile warmups to distinguish between overhead-bound and compute-bound execution.
* **Memory Discipline:** Managed computational graph state during training, explicitly detaching accumulated metrics to prevent memory leaks and optimize GPU VRAM usage per epoch.

### Acknowledgements: 
* I used AI tools to polish this readme since I wanted to brainstorm how much narrowing down the explanations would be decent.
