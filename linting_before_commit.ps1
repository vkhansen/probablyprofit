# Format code
# Lint and fix
ruff check . --fix

# Format code
black .
isort .

# Check types
mypy probablyprofit/ --config-file pyproject.toml
