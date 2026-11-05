$ErrorActionPreference = 'Stop'

$workspaceRoot = Split-Path -Parent $PSScriptRoot
$environmentPython = Join-Path $workspaceRoot '.venv\Scripts\python.exe'
$python = if (Test-Path $environmentPython) { $environmentPython } else { 'python' }
$entryPoint = Join-Path $workspaceRoot 'main.py'

& $python $entryPoint
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
