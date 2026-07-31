#!/bin/bash
uv run --no-sync scripts/run.py --config configs/apt.yaml --data.all True --data.kshot 1 --data.seeds 1
uv run --no-sync scripts/run.py --config configs/apt.yaml --data.all True --data.kshot 1 --data.seeds 10
uv run --no-sync scripts/run.py --config configs/apt.yaml --data.all True --data.kshot 1 --data.seeds 100
uv run --no-sync scripts/run.py --config configs/apt.yaml --data.all True --data.kshot 1 --data.seeds 1000
uv run --no-sync scripts/run.py --config configs/apt.yaml --data.all True --data.kshot 1 --data.seeds 10000

uv run --no-sync scripts/run.py --config configs/apt.yaml --data.all True --data.kshot 2 --data.seeds 1
uv run --no-sync scripts/run.py --config configs/apt.yaml --data.all True --data.kshot 2 --data.seeds 10
uv run --no-sync scripts/run.py --config configs/apt.yaml --data.all True --data.kshot 2 --data.seeds 100
uv run --no-sync scripts/run.py --config configs/apt.yaml --data.all True --data.kshot 2 --data.seeds 1000
uv run --no-sync scripts/run.py --config configs/apt.yaml --data.all True --data.kshot 2 --data.seeds 10000

uv run --no-sync scripts/run.py --config configs/apt.yaml --data.all True --data.kshot 4 --data.seeds 1
uv run --no-sync scripts/run.py --config configs/apt.yaml --data.all True --data.kshot 4 --data.seeds 10
uv run --no-sync scripts/run.py --config configs/apt.yaml --data.all True --data.kshot 4 --data.seeds 100
uv run --no-sync scripts/run.py --config configs/apt.yaml --data.all True --data.kshot 4 --data.seeds 1000
uv run --no-sync scripts/run.py --config configs/apt.yaml --data.all True --data.kshot 4 --data.seeds 10000

uv run --no-sync scripts/run.py --config configs/apt.yaml --data.all True --data.kshot 8 --data.seeds 1
uv run --no-sync scripts/run.py --config configs/apt.yaml --data.all True --data.kshot 8 --data.seeds 10
uv run --no-sync scripts/run.py --config configs/apt.yaml --data.all True --data.kshot 8 --data.seeds 100
uv run --no-sync scripts/run.py --config configs/apt.yaml --data.all True --data.kshot 8 --data.seeds 1000
uv run --no-sync scripts/run.py --config configs/apt.yaml --data.all True --data.kshot 8 --data.seeds 10000

uv run --no-sync scripts/run.py --config configs/apt.yaml --data.all True --data.kshot 16 --data.seeds 1
uv run --no-sync scripts/run.py --config configs/apt.yaml --data.all True --data.kshot 16 --data.seeds 10
uv run --no-sync scripts/run.py --config configs/apt.yaml --data.all True --data.kshot 16 --data.seeds 100
uv run --no-sync scripts/run.py --config configs/apt.yaml --data.all True --data.kshot 16 --data.seeds 1000
uv run --no-sync scripts/run.py --config configs/apt.yaml --data.all True --data.kshot 16 --data.seeds 10000
