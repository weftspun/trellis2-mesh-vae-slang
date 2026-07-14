# trellis2-mesh-vae-slang

Portable **Slang + Lean 4** reimplementation of the TRELLIS.2 mesh-VAE **encoder
forward** — `mesh → structured N×32 SLAT tokens` (no pooling) — replacing the
custom-CUDA sparse-conv VAE (`FlexiDualGridVaeEncoder`, `SparseConvNeXtBlock3d`,
`flex_gemm`/`spconv`) with kernels that run on any GPU (Vulkan / D3D / Metal) or CPU
and are formally verifiable.

## Why

The reference CUDA VAE works (WSL2 Linux) but is NVIDIA-only, torch-2.8/cu128-pinned
binary DLL-hell, and off-stack for V-Sekai (Godot/Vulkan runtime, Lean 4 method). We
need only the **encoder forward** for content embeddings — not training, not the
decoder/renderer.

## Approach

- Kernels (sparse-conv / ResBlock / VAE bottleneck) authored and verified in **Lean 4**
  via [`V-Sekai-fire/lean-slang`](https://github.com/V-Sekai-fire/lean-slang) (MIT),
  which builds an in-memory Slang AST and emits Slang source (`slangc -target spirv`).
- One Slang source → SPIR-V / HLSL / Metal / CPU, with built-in auto-diff.

Output SLAT tokens feed the mesh slot of
[`unified-modal-embedder`](https://github.com/weftspun/unified-modal-embedder) and,
optionally, the two-stage **ResidualFSQ-VAE with inverse-rendering** (multi-view
differentiable rendering) for geometry-faithful semantic IDs.

## Upstream decisions

In [`weftspun/multimodal-semantic-ids`](https://github.com/weftspun/multimodal-semantic-ids):
`slang-lean4-mesh-vae-over-trellis2-cuda`, `inverse-rendering-residual-fsq-vae-for-mesh-ids`.
Supersedes the archived `trellis-slat-fsq` / `slat-semantic-ids` (CUDA path).

## License

MIT © 2026 K. S. Ernest (iFire) Lee
