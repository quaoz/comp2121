default:
    just --list

jupyter:
    uv run --with jupyter jupyter lab

lint:
    ruff check --select I --fix
    ruff check --fix
    ruff format
