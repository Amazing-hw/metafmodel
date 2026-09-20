[CmdletBinding()]
param(
    [string]$ProjectRoot = 'D:\Fmodel\metafmodel',
    [string]$CanonicalRoot = 'D:\Fmodel\emg_data_canonical',
    [string]$RunsRoot = 'D:\Fmodel\runs',
    [string]$SplitsPath = '',
    [ValidateSet('Official', 'StrictSubject')][string]$Protocol = '',
    [Nullable[int]]$SplitSeed = $null,
    [ValidateSet('supervised_tcn', 'bendr', 'cbramod', 'aemg')]
    [string]$Model = 'bendr',
    [ValidateSet('pretrain', 'finetune')][string]$Stage = 'finetune',
    [ValidateSet('pose', 'wrist_angle', 'discrete_gesture', 'handwriting', 'typing')]
    [string]$Task = 'pose',
    [ValidateSet('full', 'partial', 'probe')][string]$FinetuneMode = 'full',
    [ValidateRange(0.000001, 1.0)][double]$LabelBudget = 1.0,
    [ValidateRange(0, 2147483647)][int]$UnfrozenBlocks = 1,
    [int]$Seed = 0,
    [ValidateSet('a100_ddp', 'cpu_smoke')][string]$Trainer = 'a100_ddp',
    [string]$PretrainedCheckpoint = '',
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

if ($Stage -eq 'pretrain' -and $Model -eq 'supervised_tcn') {
    throw 'supervised_tcn is supervised-only and cannot run with stage=pretrain'
}

$ProjectRoot = Resolve-FullPath -Path $ProjectRoot
$CanonicalRoot = Resolve-FullPath -Path $CanonicalRoot
$RunsRoot = Resolve-FullPath -Path $RunsRoot
$SplitsPath = Resolve-SplitManifest -CanonicalRoot $CanonicalRoot -SplitsPath $SplitsPath -Protocol $Protocol -SplitSeed $SplitSeed -AllowMissing:$AllowMissingSplit
$splitSlug = Get-SplitRunSlug -SplitsPath $SplitsPath -Protocol $Protocol -SplitSeed $SplitSeed

$entryPoint = Join-Path $ProjectRoot 'run_baseline.py'
Assert-File -Path $entryPoint -Label 'run_baseline.py'
Assert-Directory -Path $CanonicalRoot -Label 'canonical data root'
if (-not $AllowMissingSplit) { Assert-File -Path $SplitsPath -Label 'split manifest' }

if (-not [string]::IsNullOrWhiteSpace($PretrainedCheckpoint)) {
    $PretrainedCheckpoint = Resolve-FullPath -Path $PretrainedCheckpoint
    if (-not $DryRun) {
        Assert-File -Path $PretrainedCheckpoint -Label 'pretrained checkpoint'
    }
}
if (-not [string]::IsNullOrWhiteSpace($ResumeFrom)) {
    $ResumeFrom = Resolve-FullPath -Path $ResumeFrom
    if (-not $DryRun) {
        Assert-File -Path $ResumeFrom -Label 'resume checkpoint'
    }
}

$outputDirectory = Join-Path $RunsRoot "baselines\$splitSlug\$Model\$Stage\$Task\seed-$Seed"
Assert-OutputDirectoryAvailable -Path $outputDirectory -Force:$Force -DryRun:$DryRun
$arguments = @(
    $entryPoint,
    "model=$Model",
    "stage=$Stage",
    "task=$Task",
    "trainer=$Trainer",
    "data.root=$CanonicalRoot",
    "data.manifest_path=$SplitsPath",
    "seed=$Seed",
    "label_budget=$LabelBudget",
    "finetune_mode=$FinetuneMode",
    "unfrozen_blocks=$UnfrozenBlocks",
    "hydra.run.dir=$outputDirectory"
)
if (-not [string]::IsNullOrWhiteSpace($PretrainedCheckpoint)) {
    $arguments += "pretrained_checkpoint=$PretrainedCheckpoint"
}
if (-not [string]::IsNullOrWhiteSpace($ResumeFrom)) {
    $arguments += "resume_from=$ResumeFrom"
}
Invoke-Checked -Executable $Python -Arguments $arguments -DryRun:$DryRun
