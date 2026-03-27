# Pyproject patch

Use this when migrating to the `src/` layout.

## Poetry package discovery

```toml
[tool.poetry]
packages = [{ include = "chessquant_ml", from = "src" }]
```

## Optional script entries

```toml
[tool.poetry.scripts]
chessquant-ml = "chessquant_ml.cli.main:app"
```

## Dev dependencies you likely want

```toml
[tool.poetry.group.dev.dependencies]
pytest = "^8.0.0"
ruff = "^0.11.0"
black = "^25.0.0"
```
