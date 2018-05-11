# SOAPY
This is an *analytical* implementation of SOAP (Smooth Overlap of Atomic Positions) [ArXiv](https://arxiv.org/abs/1209.3140).

Since it's an exact implementation, there is no need to worry about basis functions, numerical integration or fitting errors. However, the analytical implementation scales as O(N^2) with N structures in contrast to O(N) for numerical implementations. This implementation uses `numpy` and `numba` to speed things up.

### TODO

- [x] Average kernel
- [ ] Zero out kernels of different species
- [ ] Rematch kernel
- [ ] Plot with performance compared to the original numerical implementation.
- [ ] Exact implementation of bispectrum.
