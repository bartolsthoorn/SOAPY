# SOAPY
This is an *analytical* implementation of SOAP (Smooth Overlap of Atomic Positions) [ArXiv](https://arxiv.org/abs/1209.3140).

The analytical implementation scales as O(N^2) with N structures in contrast to O(N) for numerical implementations. However, this implementation uses `numpy` and `numba` to speed things up.

### TODO

- [x] Average kernel
- [ ] Zero out kernels of different species
- [ ] Rematch kernel
- [ ] Plot with performance compared to the original numerical implementation.
- [ ] Exact implementation of bispectrum.
