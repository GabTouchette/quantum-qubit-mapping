# Enhanced SABRE Algorithm for Qubit Mapping on NISQ Devices

## Project Overview

This project enhances the SABRE (SWAP-based BidiREctional heuristic search) algorithm for qubit mapping on NISQ devices, based on the paper ["Tackling the Qubit Mapping Problem for NISQ-Era Quantum Devices"](https://arxiv.org/pdf/1809.02573.pdf) by Li, Ding, and Xie. The original implementation is based on the code from (https://github.com/Kaustuvi/quantum-qubit-mapping). The implementation now includes several critical improvements for better performance and reliability.

## Key Enhancements

1. **Physical Qubit Mapping**
   - Fixed logical-to-physical qubit translation in final circuit output
   - Added helper methods for consistent qubit mapping throughout execution

2. **Optimized SWAP Selection**
   - Implemented single best-SWAP selection per iteration
   - Reduced unnecessary SWAP operations that caused mapping thrashing

3. **Improved Heuristic Scoring**
   - Fixed decay parameter implementation
   - Enhanced scoring mechanism for more optimal SWAP selection

4. **Robust Benchmarking**
   - Added timeout mechanism for heuristic evaluation
   - Comprehensive performance metrics collection
   - Automated report generation with visualizations

5. **Verification Tools**
   - Enhanced circuit validation methods
   - Added assertion checks for physical qubit constraints
