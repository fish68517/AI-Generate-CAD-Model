param(
    [switch]$RunInference
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python environment not found: $Python"
}

Set-Location -LiteralPath $ProjectRoot
$env:HF_HOME = Join-Path $ProjectRoot '.cache\huggingface'
$env:HF_HUB_DISABLE_PROGRESS_BARS = '1'
$env:USE_TF = '0'

& $Python '.\scripts\check_environment.py'
if ($LASTEXITCODE -ne 0) { throw 'Environment check failed.' }

& $Python '.\scripts\download_sample.py'
if ($LASTEXITCODE -notin @(0, 2)) { throw 'Dataset sampling failed.' }

& $Python '.\scripts\audit_dataset.py' '--max-length' '768'
if ($LASTEXITCODE -ne 0) { throw 'Dataset audit failed.' }

& $Python '-m' 'unittest' 'discover' '-s' 'tests' '-v'
if ($LASTEXITCODE -ne 0) { throw 'Unit tests failed.' }

& $Python '.\scripts\reconstruct_json.py'
if ($LASTEXITCODE -ne 0) { throw 'CPU reconstruction failed. See reports\reconstruction\summary.json.' }

& $Python '.\scripts\generate_example_gallery.py'
if ($LASTEXITCODE -ne 0) { throw 'Example gallery generation failed.' }

& $Python '.\scripts\prepare_lora_subset.py' `
    '--tokenizer' '.\models\Qwen2.5-Coder-1.5B-Instruct' `
    '--max-length' '4096'
if ($LASTEXITCODE -ne 0) { throw 'Small-sample LoRA data preparation failed.' }

& $Python '.\scripts\infer_batch_gpu.py' '--dry-run' '--limit' '20'
if ($LASTEXITCODE -ne 0) { throw 'GPU batch inference dry-run failed.' }

& $Python '.\scripts\train_cloud.py' '--dry-run'
if ($LASTEXITCODE -ne 0) { throw 'Cloud training configuration dry-run failed.' }

& $Python '.\scripts\infer_low_resource.py' '--mode' 'dry-run' '--max-input-tokens' '768'
if ($LASTEXITCODE -ne 0) { throw 'Inference dry-run failed.' }

if ($RunInference) {
    $Base = '.\models\Qwen2.5-Coder-1.5B-Instruct'
    $Adapter = '.\models\CADmium-1.5B'
    $BaseWeights = Join-Path $Base 'model.safetensors'
    $AdapterWeights = Join-Path $Adapter 'adapter_model.safetensors'
    if (-not (Test-Path -LiteralPath $BaseWeights) -or -not (Test-Path -LiteralPath $AdapterWeights)) {
        throw 'Local 1.5B base model or adapter is missing.'
    }
    & $Python '.\scripts\infer_low_resource.py' `
        '--mode' 'cpu-fp32' `
        '--adapter' $Adapter `
        '--base-model' $Base `
        '--prompt-file' '.\data\samples\prompts\0026_00267624.txt' `
        '--max-input-tokens' '1024' `
        '--max-new-tokens' '384' `
        '--output-dir' '.\reports\inference\sample_0026_00267624'
    if ($LASTEXITCODE -ne 0) { throw 'CPU model inference failed.' }
}

Write-Host 'Local lightweight reproduction pipeline completed.' -ForegroundColor Green
