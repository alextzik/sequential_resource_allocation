# receding_resource_allocation

Receding-horizon resource allocation with Gaussian demand chance constraints (Algorithm 2 implementation).

Quick start
- Install deps (poetry):
```bash
poetry install
```
- Run the Gaussian MPC example:
```bash
poetry run python scripts/run_algorithm2_gaussian.py --resources 3 --demands 5 --horizon 12 --prob 0.9
```

Arguments
- `--resources`: number of resource types (rows in allocation matrix)
- `--demands`: number of demand classes (columns in allocation matrix)
- `--horizon`: planning horizon (time steps)
- `--prob`: chance-constraint service probability p (uses z = Phi^{-1}(p))
- `--seed`: RNG seed for synthetic problem generation