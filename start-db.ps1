# Check for virtual environment
$venvPython = ".\venv\Scripts\python.exe"

if (Test-Path $venvPython) {
    Write-Host "Using virtual environment python: $venvPython"
    $python = $venvPython
} else {
    Write-Warning "Virtual environment not found. Trying global python..."
    $python = "python"
}

# Run alembic upgrade head
Write-Host "Running database migrations (alembic upgrade head)..."
try {
    # Using python -m alembic is more reliable than calling alembic.exe directly
    & $python -m alembic upgrade head
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Database upgraded successfully!" -ForegroundColor Green
    } else {
        Write-Error "Alembic upgrade failed with exit code $LASTEXITCODE"
        exit $LASTEXITCODE
    }
} catch {
    Write-Error "Failed to execute alembic: $_"
    exit 1
}
