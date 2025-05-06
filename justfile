default:
    just --list

jupyter:
    uv run --with jupyter jupyter lab

format:
    ruff format
