# Check if global Python is installed
try {
    $globalPython = python --version 2>&1
    Write-Host "Found Global Python: $globalPython"
}
catch {
    Write-Error "Python is not installed or not in the PATH."
    exit 1
}

# Create virtual environment
if (-not (Test-Path "venv")) {
    Write-Host "Creating virtual environment..."
    python -m venv venv
} else {
    Write-Host "Virtual environment already exists."
}

# Define venv python path
$venvPython = ".\venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Error "Virtual environment python not found at $venvPython. Venv creation might have failed."
    exit 1
}

# Activate virtual environment (optional for script execution if we use direct path, but good for user awareness)
# We won't rely on activation for the script commands, but we'll use direct path to ensure venv is used.
Write-Host "Targeting virtual environment python at $venvPython"

# Upgrade pip
Write-Host "Upgrading pip..."
& $venvPython -m pip install --upgrade pip

# Check for Git
try {
    git --version | Out-Null
} catch {
    Write-Error "Git is not installed or not in PATH. Required for pulling source."
    exit 1
}

# Pull/Clone probablyprofit source
$repoUrl = "https://github.com/randomness11/probablyprofit.git"
$sourceDir = "probablyprofit-source"

if (Test-Path $sourceDir) {
    Write-Host "Updating probablyprofit source in $sourceDir..."
    Push-Location $sourceDir
    try {
        git pull
    } catch {
        Write-Warning "Failed to pull updates. You might have local changes or network issues."
    }
    Pop-Location
} else {
    Write-Host "Cloning probablyprofit from $repoUrl into $sourceDir..."
    try {
        git clone $repoUrl $sourceDir
    } catch {
        Write-Error "Failed to clone repository."
        exit 1
    }
}

# Install dependencies
Write-Host "Installing dependencies..."

# Install root requirements if they exist, filtering out probablyprofit to strictly use source
if (Test-Path "requirements.txt") {
    Write-Host "Installing dependencies from root requirements.txt (excluding probablyprofit)..."
    # Filter out probablyprofit lines to avoid PyPI installation
    Get-Content requirements.txt | Where-Object { $_ -notmatch "^probablyprofit" } | Set-Content requirements.temp.txt
    try {
        & $venvPython -m pip install -r requirements.temp.txt
    } finally {
        if (Test-Path requirements.temp.txt) {
            Remove-Item requirements.temp.txt
        }
    }
}

# Install probablyprofit from source (editable mode) with [full] extras
if (Test-Path "$sourceDir\pyproject.toml") {
    Write-Host "Installing probablyprofit library (with [full] extras) in editable mode..."
    # Quotes are important for brackets in paths/extras
    & $venvPython -m pip install -e "${sourceDir}[full]"
} elseif (Test-Path "$sourceDir\setup.py") {
    Write-Host "Installing probablyprofit library in editable mode (setup.py)..."
    & $venvPython -m pip install -e $sourceDir
} elseif (Test-Path "$sourceDir\requirements.txt") {
    Write-Host "Installing probablyprofit requirements..."
    & $venvPython -m pip install -r "$sourceDir\requirements.txt"
} else {
    Write-Warning "No installable configuration (pyproject.toml, setup.py, or requirements.txt) found in $sourceDir."
}

# Setup .env file
if (-not (Test-Path ".env")) {
    if (Test-Path ".env.example") {
        Write-Host "Creating .env from .env.example..."
        Copy-Item ".env.example" -Destination ".env"
    } else {
        Write-Warning ".env.example not found, skipping .env creation."
    }
} else {
    Write-Host ".env already exists."
}

Write-Host "Setup complete! To activate the environment, run: .\venv\Scripts\Activate.ps1"
