# Repository Guidelines

## Project Structure & Module Organization

- `src/`: core Python code (data pipeline, models, RL environment, evaluation, live bridge).
- `scripts/`: runnable entry points (`scripts/core/` for pipeline + MT5 bridge; others for analysis/visualization).
- `config/`: configuration (dataclass-based) in `config/settings.py`.
- `tests/`: pytest suite, including training ↔ live-bridge parity tests.
- `frontend/`: optional React/Vite dashboard (Tailwind + ESLint).
- Generated artifacts (gitignored): `data/`, `models/`, `results/`, `logs/`, `tmp/`, `frontend/dist/`, `frontend/node_modules/`.

## Build, Test, and Development Commands

Python setup (repo root):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

- Full pipeline (train + backtest): `python scripts/core/run_pipeline.py`
- Live MT5 bridge: `python scripts/core/prepare_mt5_bridge_artifacts.py` then `python scripts/core/run_mt5_bridge.py --host 127.0.0.1 --port 5555 --symbol US30`
- Frontend (optional): `cd frontend && npm install && npm run dev`

## Coding Style & Naming Conventions

- Python: 4-space indentation, type hints expected, and `logging.getLogger(__name__)` for module loggers.
- Performance: prefer `float32` tensors/arrays (Apple Silicon/MPS); avoid silent `float64` promotion.
- Naming: `snake_case.py` for modules; `PascalCase` classes; role-based filenames like `train_*.py`, `*_env.py`, `*_agent.py`.
- Frontend: keep `npm run lint` (ESLint) clean before opening a PR.

## Testing Guidelines

- Framework: pytest (see `pytest.ini`); tests are `tests/test_*.py`.
- Run all tests: `pytest tests/`
- Skip slow tests: `pytest tests/ -m "not slow"`
- Parity checks: `pytest tests/test_parity.py -v`

## Commit & Pull Request Guidelines

- Commit messages commonly use conventional prefixes: `feat:`, `fix:`, `docs:`, `refactor:` (versioned snapshots like `v27:` also appear).
- PRs should include: a clear summary, how you validated (tests and/or backtest metrics), and screenshots for `frontend/` changes.
- Avoid committing large artifacts; prefer reproducible pipelines and document any required data/model inputs in `README.md`.

## Security & Configuration Tips

- Keep credentials and broker/account details out of git; use env vars (e.g., `US30_DATA_DIR`) or local `.env` files.
- For live trading, start with `--dry-run` and verify symbol/timezone/resampling parity before enabling execution.
