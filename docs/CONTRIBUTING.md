# Contributing to VPR-SNN

Thanks for your interest in contributing. This project is research-focused; please follow these guidelines to keep contributions clear and reproducible.

- Issues
  - Open an issue for bugs, enhancement requests, or reproducibility questions.
  - Provide a minimal reproduction when reporting bugs (code snippet, PyTorch + snntorch versions).

- Pull Requests
  - Fork -> feature branch -> PR against `main` (or `master`, depending on your remote).
  - Keep changes small and focused: one logical change per PR.
  - Include tests or a short script demonstrating your change when possible.
  - Write a concise PR description and mention related issues.

- Code Style
  - Follow existing repository style (PEP8 for Python).
  - Use descriptive names; avoid one-letter variables.
  - Do not change unrelated code formatting.

- Tests
  - There are no formal tests in the repo yet. If you add behavior-critical changes, please include a small test script under `tests/`.

- Licensing
  - This repository includes a `LICENSE` file. By contributing, you agree to license your contributions under the same terms.

If you'd like, I can add a `scripts/` folder with small runnable examples (train/eval-only), or scaffold a `tests/` folder for CI. Ask and I'll add it.
