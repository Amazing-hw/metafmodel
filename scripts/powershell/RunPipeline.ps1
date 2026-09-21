[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Stages,
    [string]$ProjectRoot = 'D:\Fmodel\metafmodel',
    [string]$CanonicalRoot = 'D:\Fmodel\emg_data_canonical',
    [string]$RunsRoot = 'D:\Fmodel\runs',
    [string]$SplitsPath = '',
    [ValidateSet('Official', 'StrictSubject')][string]$Protocol = '',
    [Nullable[int]]$SplitSeed = $null,
    [string]$CheckpointPath = '',
    [switch]$Force,
    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'Common.ps1')

$ProjectRoot = [IO.Path]::GetFullPath($ProjectRoot)
$CanonicalRoot = [IO.Path]::GetFullPath($CanonicalRoot)
$RunsRoot = [IO.Path]::GetFullPath($RunsRoot)
$stageMap = @{
    Preflight = '00_Preflight.ps1'; Audit = '01_AuditCanonical.ps1'; Splits = '02_PrepareSplits.ps1';
    Smoke = '03_SmokeTest.ps1'; Baseline = '04_TrainBaseline.ps1'; Pretrain = '05_Pretrain.ps1';
    Finetune = '06_Finetune.ps1'; Evaluate = '07_Evaluate.ps1'; Robustness = '08_Robustness.ps1'
}
$splitConsumerStages = @('Smoke', 'Baseline', 'Pretrain', 'Finetune', 'Evaluate', 'Robustness')
$selected = @($Stages -split ',' | ForEach-Object { $_.Trim() } | Where-Object { $_ })
if ($selected.Count -eq 0) { throw 'Stages must contain at least one stage name' }
foreach ($stage in $selected) {
    if (-not $stageMap.ContainsKey($stage)) { throw "unknown stage: $stage" }
}
$splitStageIndex = [Array]::IndexOf($selected, 'Splits')
$consumerIndices = @(
    for ($index = 0; $index -lt $selected.Count; $index++) {
        if ($selected[$index] -in $splitConsumerStages) { $index }
    }
)
if ($splitStageIndex -ge 0 -and $consumerIndices.Count -gt 0 -and $splitStageIndex -gt ($consumerIndices | Measure-Object -Minimum).Minimum) {
    throw 'Splits must precede every split-consuming stage in -Stages'
}

$needsSplit = @($selected | Where-Object { $_ -eq 'Splits' -or $_ -in $splitConsumerStages }).Count -gt 0
$selectedSplitPath = ''
$usingExplicitSplitsPath = -not [string]::IsNullOrWhiteSpace($SplitsPath)
if ($usingExplicitSplitsPath -and (-not [string]::IsNullOrWhiteSpace($Protocol) -or $null -ne $SplitSeed)) {
    throw 'cannot combine -SplitsPath with -Protocol or -SplitSeed'
}
if ($needsSplit) {
    if ($usingExplicitSplitsPath) {
        if ($selected -contains 'Splits') {
            throw 'cannot combine -SplitsPath with the Splits stage; select -Protocol and -SplitSeed instead'
        }
        $selectedSplitPath = Resolve-SplitManifest -CanonicalRoot $CanonicalRoot -SplitsPath $SplitsPath
    }
    else {
        if ([string]::IsNullOrWhiteSpace($Protocol) -or $null -eq $SplitSeed) {
            throw 'split selection requires -SplitsPath or both -Protocol and -SplitSeed'
        }
        $splitSlug = if ($Protocol -eq 'Official') { 'official' } else { 'strict-subject' }
        $selectedSplitPath = Join-Path $CanonicalRoot "splits.$splitSlug.seed-$SplitSeed.csv"
    }
}

foreach ($stage in $selected) {
    $scriptPath = Join-Path $PSScriptRoot $stageMap[$stage]
    if (-not (Test-Path -LiteralPath $scriptPath -PathType Leaf)) { throw "stage script not found: $scriptPath" }
    $arguments = @{
        ProjectRoot = $ProjectRoot
        CanonicalRoot = $CanonicalRoot
    }
    if ($stage -ne 'Splits') { $arguments.RunsRoot = $RunsRoot }
    if ($stage -eq 'Splits') {
        $arguments.Protocol = $Protocol
        $arguments.Seed = $SplitSeed
        if ($Force) { $arguments.Force = $true }
    }
    elseif ($stage -in $splitConsumerStages) {
        if ($usingExplicitSplitsPath) {
            $arguments.SplitsPath = $selectedSplitPath
        }
        else {
            $arguments.Protocol = $Protocol
            $arguments.SplitSeed = $SplitSeed
        }
        if ($Force) { $arguments.Force = $true }
        $currentStageIndex = [Array]::IndexOf($selected, $stage)
        if ($DryRun -and $splitStageIndex -ge 0 -and $splitStageIndex -lt $currentStageIndex) {
            $arguments.AllowMissingSplit = $true
        }
    }
    if ($stage -in @('Evaluate', 'Robustness')) {
        if ([string]::IsNullOrWhiteSpace($CheckpointPath)) { throw "$stage requires -CheckpointPath" }
        $arguments.CheckpointPath = $CheckpointPath
    }
    if ($DryRun) { $arguments.DryRun = $true }
    Write-Host "Stage: $stage"
    & $scriptPath @arguments
    if (-not $?) { throw "stage failed: $stage" }
}
