param(
    [switch]$SkipInstall,
    [switch]$SkipDownload
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPython = Join-Path $ProjectRoot '.venv\Scripts\python.exe'

if (-not (Test-Path -LiteralPath $VenvPython)) {
    py -3.12 -m venv (Join-Path $ProjectRoot '.venv')
}

if (-not $SkipInstall) {
    & $VenvPython -m pip install --upgrade pip
    & $VenvPython -m pip install -r (Join-Path $ProjectRoot 'requirements-local.txt')
}

$PipelineArgs = @()
if ($SkipDownload) {
    $PipelineArgs += '--skip-download'
}

& $VenvPython (Join-Path $ProjectRoot 'scripts\run_local_pipeline.py') @PipelineArgs
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

& $VenvPython -m pytest (Join-Path $ProjectRoot 'tests') -q
exit $LASTEXITCODE
