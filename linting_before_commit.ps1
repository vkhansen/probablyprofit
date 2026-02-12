param(
    [switch]$unsafe
)

# Format code
# Lint and fix
if ($unsafe) {
    Write-Host "Running with unsafe fixes..."
    ruff check . --fix --unsafe-fixes
} else {
    ruff check . --fix
}

# Format code
black .
isort .

# Check types
mypy probablyprofit/ --config-file pyproject.toml
