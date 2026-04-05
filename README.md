# Spine: Non-Linear State Management for Sequence Modeling

## Overview
Spine is a hybrid architecture inspired by State Space Models (SSM) and Recurrent Neural Networks (RNN). It replaces the traditional KV-cache found in Transformers with a compressed, recursive state vector called "The Spine". This approach enables constant latency and memory usage regardless of sequence length.

## Core Components
- **TheSpine Module**: A PyTorch implementation of the recursive state update with non-linear tanh evolution.
- **Symmetry Breaking**: Diverse time-scales for multi-scale memory retention.
- **Selective Gating**: Input-dependent updates to differentiate between semantic essence and noise.
- **RMS-Harness**: Per-step normalization to ensure numerical stability over long contexts.

## Technical Specifications
- Architecture: Non-linear SSM Hybrid.
- State Persistence: Verified over 50+ turns of noise (cosine similarity > 0.59).
- Complexity: O(1) inference time per token relative to history length.

## Usage
Refer to `spine.py` for the model implementation and `test_memory.py` for stability verification.
