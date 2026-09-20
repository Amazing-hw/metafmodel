[CmdletBinding()]
param(
    [string]$ProjectRoot = 'D:\Fmodel\metafmodel',
    [string]$CanonicalRoot = 'D:\Fmodel\emg_data_canonical',
    [string]$RunsRoot = 'D:\Fmodel\runs',
    [string]$SplitsPath = '',
    [ValidateSet('Official', 'StrictSubject')][string]$Protocol = '',
    [Nullable[int]]$SplitSeed = $null,
    [ValidateSet('Pilot', 'Full', 'Resume')][string]$Mode = 'Pilot',
    [int]$MaxSteps = 0,
    [ValidateRange(1, 2147483647)][int]$Devices = 1,
    [ValidateRange(1, 2147483647)][int]$AccumulateGradBatches = 1,
    [ValidateRange(1, 2147483647)][int]$BatchSize = 8,
    [ValidateSet('a100_ddp', 'cpu_smoke')][string]$Trainer = 'a100_ddp',
    [int]$Seed = 0,
    [string]$ResumeFrom = '',
    [string]$Python = 'python',
    [switch]$Force,
    [switch]$AllowMissingSplit,
    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'Common.ps1')
if ($AllowMissingSplit -and -not $DryRun) { throw '-AllowMissingSplit is only valid with -DryRun' }

if ($Mode -eq 'Pilot' -and $MaxSteps -eq 0) {
    $MaxSteps = 200
}
if ($MaxSteps -le 0) {
    throw "Mode=$Mode requires an explicit positive -MaxSteps value"
}
if ($Mode -eq 'Resume' -and [string]::IsNullOrWhiteSpace($ResumeFrom)) {
    throw 'Mode=Resume requires -ResumeFrom'
}

$ProjectRoot = Resolve-FullPath -Path $ProjectRoot
$CanonicalRoot = Resolve-FullPath -Path $CanonicalRoot
$RunsRoot = Resolve-FullPath -Path $RunsRoot
$SplitsPath = Resolve-SplitManifest -CanonicalRoot $CanonicalRoot -SplitsPath $SplitsPath -Protocol $Protocol -SplitSeed $SplitSeed -AllowMissing:$AllowMissingSplit
$splitSlug = Get-SplitRunSlug -SplitsPath $SplitsPath -Protocol $Protocol -SplitSeed $SplitSeed

$entryPoint = Join-Path $ProjectRoot 'pretrain.py'
Assert-File -Path $entryPoint -Label 'pretrain.py'
Assert-Directory -Path $CanonicalRoot -Label 'canonical data root'
if (-not $AllowMissingSplit) { Assert-File -Path $SplitsPath -Label 'split manifest' }

if (-not [string]::IsNullOrWhiteSpace($ResumeFrom)) {
    $ResumeFrom = Resolve-FullPath -Path $ResumeFrom
    if (-not $DryRun) {
        Assert-File -Path $ResumeFrom -Label 'resume checkpoint'
    }
}

$modeSlug = $Mode.ToLowerInvariant()
$outputDirectory = Join-Path $RunsRoot "pretrain\$splitSlug\$modeSlug\seed-$Seed"
Assert-OutputDirectoryAvailable -Path $outputDirectory -Force:$Force -DryRun:$DryRun
$globalBatchSize = $Devices * $AccumulateGradBatches * $BatchSize
Write-Host "mode=$Mode"
Write-Host "global_batch_size=$globalBatchSize"
Write-Host "run_directory=$outputDirectory"

$arguments = @(
    $entryPoint,
    "trainer=$Trainer",
    "trainer.devices=$Devices",
    "trainer.accumulate_grad_batches=$AccumulateGradBatches",
    "data.root=$CanonicalRoot",
    "data.manifest_path=$SplitsPath",
    "data.batch_size=$BatchSize",
    "optimizer.max_steps=$MaxSteps",
    "seed=$Seed",
    "hydra.run.dir=$outputDirectory"
)
if (-not [string]::IsNullOrWhiteSpace($ResumeFrom)) {
    $arguments += "resume_from=$ResumeFrom"
}
Invoke-Checked -Executable $Python -Arguments $arguments -DryRun:$DryRun
