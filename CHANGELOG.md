# Changelog

All notable changes to this project are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses [Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-10-25

### Added
- `SimulatedBifurcationSampler` with ballistic (`bSB`) and discrete (`dSB`) modes, seeded and reproducible.
- Numba-compiled kernel and a pure NumPy reference implementation that agree bit-for-bit.
- Input validation with explicit `ValueError`s.
- Examples: quickstart, Max-Cut and QUBO, each checked against brute force.
- Benchmarks against four classical baselines, including the losing cases.
