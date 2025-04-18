# Enhanced SABRE Algorithm for Qubit Mapping on NISQ Devices

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

This project implements an improved version of the **SWAP-based BidiREctional heuristic search algorithm (SABRE)** for qubit mapping on NISQ-era quantum devices, based on the original paper by Li, Ding, and Xie ([arXiv:1809.02573](https://arxiv.org/abs/1809.02573)). The original implementation is based on the code from (https://github.com/Kaustuvi/quantum-qubit-mapping).

## Key Improvements

### 1. Fixed Implementation Issues
- **Correct Physical Qubit Labeling**: Ensured final program uses proper physical qubit indices
- **Optimized SWAP Selection**: Now selects a single best SWAP per iteration instead of one per front-layer gate
- **Decay Parameter Fix**: Properly maintains decay values throughout execution
- **Adjacent SWAP Cancellation**: Detects and removes redundant SWAP sequences (A-B followed by B-A)

### 2. Enhanced Heuristics
Implemented and tested several heuristic approaches:
- **Critical Path Heuristic**: Prioritizes gates on the critical path
- **Entanglement-Aware Heuristic**: Considers qubit entanglement patterns
- **Lookahead Window Heuristic**: Evaluates future gate dependencies
- **Hybrid Lexicographic Heuristic (hybrid_lexi)**: Our novel state-aware approach combining:
  - Immediate gate distance (local optimization)
  - Extended successor gate distance (temporal awareness)
  - Global qubit ordering improvement (structural optimization)
