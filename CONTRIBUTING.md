# Contributing

Bug reports, numerical edge cases, and reproducibility improvements are welcome.
Please include the command, configuration, dependency versions, and the smallest
input needed to reproduce an issue. Do not attach private datasets or credentials.

## Development

```bash
python -m pip install -e '.[test]'
python -m pytest -q
python -m spatial_sharing smoke --output runs/smoke-dev
```

The upstream CBDE test is optional locally; obtain its pinned source with
`python scripts/fetch_airnet.py`. CI checks out that revision and runs the test.
The full DGRN path additionally requires CUDA-enabled MMCV and is checked with
`python scripts/check_airnet.py` on a CUDA machine.

## Research changes

- Keep output fitting independent of clean targets.
- Preserve unclipped RGB arrays for decomposition; compute clipped image metrics separately.
- Fit feature normalization on training records only, and select checkpoints using validation data.
- Record any changes to crop conventions, curve knots, coefficient bounds, or metric definitions.
- Include focused tests for changes to equations, gradients, data splits, or checkpoint behavior.
- Describe actual validation. Synthetic fixtures and smoke-backbone scores are not benchmark results.

Submit a pull request with the problem, the resulting behavior, and the checks
you ran. Contributions to the original code are made under the repository's MIT
license; separately obtained dependencies retain their own terms.
