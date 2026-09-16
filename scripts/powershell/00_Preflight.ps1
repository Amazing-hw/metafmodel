[CmdletBinding()]
param(
    [string]$ProjectRoot = 'D:\Fmodel\metafmodel',
    [string]$CanonicalRoot = 'D:\Fmodel\emg_data_canonical',
    [string]$RunsRoot = 'D:\Fmodel\runs',
    [string]$Python = 'python',
    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'Common.ps1')

$ProjectRoot = Resolve-FullPath -Path $ProjectRoot
$CanonicalRoot = Resolve-FullPath -Path $CanonicalRoot
$RunsRoot = Resolve-FullPath -Path $RunsRoot

Write-Host "ProjectRoot=$ProjectRoot"
Write-Host "CanonicalRoot=$CanonicalRoot"
Write-Host "RunsRoot=$RunsRoot"

Assert-Directory -Path $ProjectRoot -Label 'project root'
Assert-Directory -Path $CanonicalRoot -Label 'canonical data root'
Assert-File -Path (Join-Path $CanonicalRoot 'manifest.csv') -Label 'canonical manifest'

foreach ($entryPoint in @(
    'make_splits.py',
    'pretrain.py',
    'run_baseline.py',
    'finetune.py',
    'evaluate.py'
)) {
    Assert-File -Path (Join-Path $ProjectRoot $entryPoint) -Label $entryPoint
}

if (-not $DryRun) {
    $RunsRoot = Ensure-OutputDirectory -Path $RunsRoot
}

$probe = @(
    '-c',
    'import torch, lightning, hydra, omegaconf, pandas, h5py; print(torch.__version__); print(torch.cuda.is_available())'
)
Invoke-Checked -Executable $Python -Arguments $probe -DryRun:$DryRun
