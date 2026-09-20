[CmdletBinding()]
param(
    [string]$ProjectRoot = 'D:\Fmodel\metafmodel',
    [string]$CanonicalRoot = 'D:\Fmodel\emg_data_canonical',
    [string]$RunsRoot = 'D:\Fmodel\runs',
    [string]$SplitsPath = '',
    [ValidateSet('Official', 'StrictSubject')][string]$Protocol = '',
    [Nullable[int]]$SplitSeed = $null,
    [ValidateSet('All', 'pose', 'wrist_angle', 'discrete_gesture', 'handwriting', 'typing')]
    [string]$Task = 'All',
    [int]$Seed = 0,
    [string]$Python = 'python',
    [switch]$Force,
    [switch]$AllowMissingSplit,
    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'Common.ps1')
if ($AllowMissingSplit -and -not $DryRun) { throw '-AllowMissingSplit is only valid with -DryRun' }

$ProjectRoot = Resolve-FullPath -Path $ProjectRoot
$CanonicalRoot = Resolve-FullPath -Path $CanonicalRoot
$RunsRoot = Resolve-FullPath -Path $RunsRoot
$SplitsPath = Resolve-SplitManifest -CanonicalRoot $CanonicalRoot -SplitsPath $SplitsPath -Protocol $Protocol -SplitSeed $SplitSeed -AllowMissing:$AllowMissingSplit
$splitSlug = Get-SplitRunSlug -SplitsPath $SplitsPath -Protocol $Protocol -SplitSeed $SplitSeed

$entryPoint = Join-Path $ProjectRoot 'finetune.py'
Assert-File -Path $entryPoint -Label 'finetune.py'
Assert-Directory -Path $CanonicalRoot -Label 'canonical data root'
if (-not $AllowMissingSplit) { Assert-File -Path $SplitsPath -Label 'split manifest' }

$tasks = if ($Task -eq 'All') {
    @('pose', 'wrist_angle', 'discrete_gesture', 'handwriting', 'typing')
}
else {
    @($Task)
}

foreach ($taskName in $tasks) {
    $outputDirectory = Join-Path $RunsRoot "smoke\$splitSlug\$taskName"
    Assert-OutputDirectoryAvailable -Path $outputDirectory -Force:$Force -DryRun:$DryRun
    $arguments = @(
        $entryPoint,
        'trainer=cpu_smoke',
        "task=$taskName",
        "data.root=$CanonicalRoot",
        "data.manifest_path=$SplitsPath",
        'data.num_workers=0',
        "seed=$Seed",
        "hydra.run.dir=$outputDirectory"
    )
    Invoke-Checked -Executable $Python -Arguments $arguments -DryRun:$DryRun
}
