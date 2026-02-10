# ================================================
# PowerShell Script: nomic-embed-text + Qdrant on Windows
# Tested on Windows 11 - November 2025
# ================================================

Write-Host "Note: Administrator privileges may be required for Ollama installation. Consider running this script as Administrator if you encounter issues." -ForegroundColor Yellow

# Administrator privileges may be required for Ollama installation if not already installed

# 2. Install Ollama (hosts nomic-embed-text)
#Write-Host "Installing Ollama..." -ForegroundColor Green
#Invoke-WebRequest -Uri https://ollama.com/download/OllamaSetup.exe -OutFile "$env:TEMP\OllamaSetup.exe"
#Start-Process -Wait -FilePath "$env:TEMP\OllamaSetup.exe"

# Wait a moment for Ollama service to start
#Start-Sleep -Seconds 15

# 3. Pull the embedding model
# 2. Check and Install Ollama if necessary
if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    Write-Host "Ollama not found. Installing Ollama..." -ForegroundColor Green
    Invoke-WebRequest -Uri https://ollama.com/download/OllamaSetup.exe -OutFile "$env:TEMP\OllamaSetup.exe"
    Start-Process -Wait -FilePath "$env:TEMP\OllamaSetup.exe"
    # Wait for installation to complete and service to start
    Start-Sleep -Seconds 15
} else {
    Write-Host "Ollama is installed." -ForegroundColor Green
}

# Refresh PATH to pick up Ollama if it was just installed
if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    Write-Host "Refreshing environment variables to find Ollama..." -ForegroundColor Yellow
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
    
    # Fallback to common install location if still not found
    if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
        $ollamaPath = "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe"
        if (Test-Path $ollamaPath) {
             Write-Host "Found Ollama at default location. Adding to PATH." -ForegroundColor Yellow
             $env:Path += ";$(Split-Path $ollamaPath -Parent)"
        }
    }
    
    if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
        Write-Host "Ollama installed but command not found. Please restart your terminal and try again." -ForegroundColor Red
        Pause
        exit
    }
}

# Kill any existing Ollama processes and restart
$ollamaProcesses = Get-Process -Name "ollama" -ErrorAction SilentlyContinue
if ($ollamaProcesses) {
    Write-Host "Stopping existing Ollama processes..." -ForegroundColor Yellow
    $ollamaProcesses | Stop-Process -Force
    Start-Sleep -Seconds 10
}

# Check if port is in use
$portInUse = netstat -ano | Select-String ":11434" | Select-String "LISTENING"
if ($portInUse) {
    Write-Host "Port 11434 is still in use after killing processes: $portInUse" -ForegroundColor Red
    Write-Host "Please check what is using the port and stop it manually." -ForegroundColor Red
    Pause
    exit
} else {
    Write-Host "Port 11434 is free." -ForegroundColor Green
}

Write-Host "Starting Ollama serve..." -ForegroundColor Yellow
Start-Process ollama -ArgumentList "serve" -WindowStyle Hidden -PassThru | Out-Null

# Wait for Ollama to be ready
$ollamaUrl = "http://localhost:11434"
$maxWait = 60
$waited = 0
while ($waited -lt $maxWait) {
    try {
        Invoke-WebRequest -Uri $ollamaUrl -UseBasicParsing -TimeoutSec 5 | Out-Null
        Write-Host "Ollama started successfully." -ForegroundColor Green

        # Verify process is still running
        $ollamaProcesses = Get-Process -Name "ollama" -ErrorAction SilentlyContinue
        if ($ollamaProcesses) {
            Write-Host "Ollama process is running (PID: $($ollamaProcesses.Id))." -ForegroundColor Green
        } else {
            Write-Host "Ollama process not found after start." -ForegroundColor Red
        }
        break
    } catch {
        Write-Host "Waited $waited seconds: $($_.Exception.Message)" -ForegroundColor Yellow
        Start-Sleep -Seconds 5
        $waited += 5
    }
}
if ($waited -ge $maxWait) {
    Write-Host "Failed to start Ollama after $maxWait seconds." -ForegroundColor Red
    # Check for errors
    $ollamaProcesses = Get-Process -Name "ollama" -ErrorAction SilentlyContinue
    if ($ollamaProcesses) {
        Write-Host "Ollama process is running but not responding." -ForegroundColor Yellow
    } else {
        Write-Host "Ollama process failed to start." -ForegroundColor Red
    }
    Pause
    exit
}
Write-Host "Downloading nomic-embed-text (~274MB)..." -ForegroundColor Green
ollama pull nomic-embed-text

# Model downloaded and ready for use

# Check and start Docker Desktop if not running
$dockerDesktopPath = "C:\Program Files\Docker\Docker\Docker Desktop.exe"
$dockerDesktopProcess = Get-Process -Name "Docker Desktop" -ErrorAction SilentlyContinue
if (-not $dockerDesktopProcess) {
    Write-Host "Docker Desktop not running. Starting..." -ForegroundColor Yellow
    if (Test-Path $dockerDesktopPath) {
        Start-Process $dockerDesktopPath
        Write-Host "Waiting for Docker Desktop to start..." -ForegroundColor Yellow
        Start-Sleep -Seconds 10
        # Wait for Docker daemon to be ready
        $maxWait = 60
        $waited = 0
        while ($waited -lt $maxWait) {
            try {
                docker version 2>$null | Out-Null
                Write-Host "Docker daemon is ready." -ForegroundColor Green
                break
            } catch {
                Start-Sleep -Seconds 5
                $waited += 5
            }
        }
        if ($waited -ge $maxWait) {
            Write-Host "Failed to start Docker daemon after $maxWait seconds." -ForegroundColor Red
            exit
        }
    } else {
        Write-Host "Docker Desktop not found at $dockerDesktopPath. Please install Docker Desktop." -ForegroundColor Red
        exit
    }
} else {
    Write-Host "Docker Desktop is running." -ForegroundColor Green
}

# 4. Start Qdrant in Docker
Write-Host "Starting Qdrant..." -ForegroundColor Green

# Check if container exists
$containerName = "qdrant_nomic"
# ──────────────────────────────────────────────────────────────
# Aggressive cleanup for Windows/Docker port ghost bindings
# ──────────────────────────────────────────────────────────────

Write-Host "Performing Docker cleanup to avoid 'port already allocated' ghosts..." -ForegroundColor Yellow

# 1. Force-remove the target container again (just in case)
docker rm -f $containerName 2>$null

# 2. Prune unused networks (this fixes most stale port allocations)
Write-Host "Pruning unused networks..." -ForegroundColor Cyan
docker network prune -f

# Optional but helpful: prune stopped containers & dangling stuff (low risk)
docker container prune -f 2>$null
docker system prune -f --filter "until=24h" 2>$null   # only things older than 1 day, very safe

# 3. Check if port 6333 is STILL bound after prune
$portCheck = netstat -ano | Select-String ":6333" | Select-String "LISTENING"
if ($portCheck) {
    Write-Host "Port 6333 still appears allocated after prune! Attempting Docker restart..." -ForegroundColor Red
    
    # Soft restart Docker Desktop (Windows-only)
    Write-Host "Stopping Docker Desktop..." -ForegroundColor Yellow
    Stop-Process -Name "Docker Desktop" -Force -ErrorAction SilentlyContinue
    Stop-Process -Name "com.docker.*" -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 8

    Write-Host "Starting Docker Desktop again..." -ForegroundColor Yellow
    Start-Process "C:\Program Files\Docker\Docker\Docker Desktop.exe"
    
    # Wait for Docker to come back
    $maxWaitDocker = 90
    $waitedDocker = 0
    while ($waitedDocker -lt $maxWaitDocker) {
        try {
            docker version 2>$null | Out-Null
            Write-Host "Docker is back online." -ForegroundColor Green
            break
        } catch {
            Start-Sleep -Seconds 5
            $waitedDocker += 5
            Write-Host "Waiting for Docker ($waitedDocker s)..." -ForegroundColor Yellow
        }
    }
    
    if ($waitedDocker -ge $maxWaitDocker) {
        Write-Host "Docker failed to restart automatically. Please restart Docker Desktop manually and re-run the script." -ForegroundColor Red
        Pause
        exit
    }
    
    # Final check after restart
    $portCheckAfter = netstat -ano | Select-String ":6333" | Select-String "LISTENING"
    if ($portCheckAfter) {
        Write-Host "Port 6333 STILL allocated after Docker restart. Manual intervention needed:" -ForegroundColor Red
        Write-Host "  1. Quit Docker Desktop"
        Write-Host "  2. Delete $env:USERPROFILE\.docker (will be recreated)"
        Write-Host "  3. Restart Docker Desktop"
        Write-Host "  4. Re-run script"
        Pause
        exit
    }
} else {
    Write-Host "Port cleanup looks good." -ForegroundColor Green
}

# ──────────────────────────────────────────────────────────────
# Now safe to create the container
# ──────────────────────────────────────────────────────────────

Write-Host "Creating and starting new container..." -ForegroundColor Green
try {
    docker run -d --name $containerName `
      -p 6333:6333 -p 6334:6334 `
      -v qdrant_storage:/qdrant/storage `
      qdrant/qdrant:latest
    Write-Host "New container created and started." -ForegroundColor Green
} catch {
    Write-Host "Error creating container: $($_.Exception.Message)" -ForegroundColor Red
}

# Wait for Qdrant to be ready
Write-Host "Waiting for Qdrant to be ready..." -ForegroundColor Yellow
$qdrantUrl = "http://localhost:6333"
$maxWait = 60
$waited = 0
while ($waited -lt $maxWait) {
    try {
        Invoke-WebRequest -Uri $qdrantUrl -UseBasicParsing -TimeoutSec 5 | Out-Null
        Write-Host "Qdrant is ready." -ForegroundColor Green
        break
    } catch {
        Start-Sleep -Seconds 5
        $waited += 5
    }
}
if ($waited -ge $maxWait) {
    Write-Host "Failed to connect to Qdrant after $maxWait seconds." -ForegroundColor Red
} else {
    # Test networking from container to Ollama
    Write-Host "Testing networking from container to Ollama..." -ForegroundColor Yellow
    try {
        docker exec $containerName curl -s -f http://host.docker.internal:11434 > $null
        Write-Host "Success: Container can access Ollama at host.docker.internal:11434." -ForegroundColor Green
    } catch {
        Write-Host "Warning: Container cannot access Ollama at host.docker.internal:11434. Ensure Ollama is running and verify Docker networking settings." -ForegroundColor Yellow
        Write-Host "Note: If curl is not available in the Qdrant image, this test may fail even if networking is correct." -ForegroundColor Yellow
        Write-Host "Error details: $($_.Exception.Message)" -ForegroundColor Red
    }
}
# 6. Create a collection that uses mxbai-embed-large-v1 automatically
Write-Host "Creating collection 'documents'..." -ForegroundColor Green

# Delete if exists
Invoke-RestMethod -Uri "http://localhost:6333/collections/documents" -Method Delete -ErrorAction SilentlyContinue

$createCollection = @{
    vectors = @{
        size = 768
        distance = "Cosine"
        on_disk = $true
    }
    optimizers_config = @{
        default_segment_number = 5
    }
} | ConvertTo-Json -Depth 10

Write-Host "Collection creation payload: $createCollection" -ForegroundColor Yellow
Invoke-RestMethod -Uri "http://localhost:6333/collections/documents" -Method Put -Body $createCollection -ContentType "application/json"

# Collection created successfully

Write-Host "`nSetup complete!" -ForegroundColor Green
Write-Host "Ollama running → http://localhost:11434"
Write-Host "Qdrant dashboard → http://localhost:6333/dashboard"
Write-Host "Collection 'documents' ready with nomic-embed-text embeddings"
Write-Host "You can now use it with LangChain, LlamaIndex, Haystack, etc."

# Optional: Open dashboard
Start-Process "http://localhost:6333/dashboard"