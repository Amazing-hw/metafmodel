[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Stages,
    [string]$ProjectRoot = 'D:\Fmodel\metafmodel',
    [string]$CanonicalRoot = 'D:\Fmodel\emg_data_canonical',
    [string]$RunsRoot = 'D:\Fmodel\runs',
    [string]$CheckpointPath = '',
    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$ProjectRoot = [IO.Path]::GetFullPath($ProjectRoot)
$CanonicalRoot = [IO.Path]::GetFullPath($CanonicalRoot)
$RunsRoot = [IO.Path]::GetFullPath($RunsRoot)
$stageMap = @{
    Preflight = '00_Preflight.ps1'; Audit = '01_AuditCanonical.ps1'; Splits = '02_PrepareSplits.ps1';
    Smoke = '03_SmokeTest.ps1'; Baseline = '04_TrainBaseline.ps1'; Pretrain = '05_Pretrain.ps1';
    Finetune = '06_Finetune.ps1'; Evaluate = '07_Evaluate.ps1'; Robustness = '08_Robustness.ps1'
}
$selected = @($Stages -split ',' | ForEach-Object { $_.Trim() } | Where-Object { $_ })
if ($selected.Count -eq 0) { throw 'Stages must contain at least one stage name' }

foreach ($stage in $selected) {
    if (-not $stageMap.ContainsKey($stage)) { throw "unknown stage: $stage" }
    $scriptPath = Join-Path $PSScriptRoot $stageMap[$stage]
    if (-not (Test-Path -LiteralPath $scriptPath -PathType Leaf)) { throw "stage script not found: $scriptPath" }
    $arguments = @{
        ProjectRoot = $ProjectRoot
        CanonicalRoot = $CanonicalRoot
    }
    if ($stage -ne 'Splits') { $arguments.RunsRoot = $RunsRoot }
    if ($stage -in @('Evaluate', 'Robustness')) {
        if ([string]::IsNullOrWhiteSpace($CheckpointPath)) { throw "$stage requires -CheckpointPath" }
        $arguments.CheckpointPath = $CheckpointPath
    }
    if ($DryRun) { $arguments.DryRun = $true }
    Write-Host "Stage: $stage"
    & $scriptPath @arguments
    if (-not $?) { throw "stage failed: $stage" }
}
