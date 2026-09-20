[CmdletBinding()]
param(
    [string]$ProjectRoot = 'D:\Fmodel\metafmodel', [string]$CanonicalRoot = 'D:\Fmodel\emg_data_canonical',
    [string]$RunsRoot = 'D:\Fmodel\runs', [string]$SplitsPath = '',
    [ValidateSet('Official','StrictSubject')][string]$Protocol = '', [Nullable[int]]$SplitSeed = $null,
    [ValidateSet('pose','wrist_angle','discrete_gesture','handwriting','typing')][string]$Task = 'pose',
    [ValidateSet('full','partial','probe')][string]$FinetuneMode = 'full',
    [ValidateRange(0.000001,1.0)][double]$LabelBudget = 1.0,
    [ValidateRange(0,2147483647)][int]$UnfrozenBlocks = 1, [int]$Seed = 0,
    [ValidateSet('a100_ddp','cpu_smoke')][string]$Trainer = 'a100_ddp',
    [string]$PretrainedCheckpoint = '', [string]$ResumeFrom = '', [string]$Python = 'python', [switch]$Force, [switch]$AllowMissingSplit, [switch]$DryRun
)
Set-StrictMode -Version Latest; $ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'Common.ps1')
if ($AllowMissingSplit -and -not $DryRun) { throw '-AllowMissingSplit is only valid with -DryRun' }
$ProjectRoot=Resolve-FullPath $ProjectRoot; $CanonicalRoot=Resolve-FullPath $CanonicalRoot; $RunsRoot=Resolve-FullPath $RunsRoot
$SplitsPath=Resolve-SplitManifest -CanonicalRoot $CanonicalRoot -SplitsPath $SplitsPath -Protocol $Protocol -SplitSeed $SplitSeed -AllowMissing:$AllowMissingSplit; $splitSlug=Get-SplitRunSlug -SplitsPath $SplitsPath -Protocol $Protocol -SplitSeed $SplitSeed
$entryPoint=Join-Path $ProjectRoot 'finetune.py'; Assert-File $entryPoint 'finetune.py'; Assert-Directory $CanonicalRoot 'canonical data root'; if (-not $AllowMissingSplit) { Assert-File $SplitsPath 'split manifest' }
foreach ($item in @(@{Value=$PretrainedCheckpoint;Label='pretrained checkpoint'},@{Value=$ResumeFrom;Label='resume checkpoint'})) {
    if (-not [string]::IsNullOrWhiteSpace($item.Value)) { $resolved=Resolve-FullPath $item.Value; if (-not $DryRun) { Assert-File $resolved $item.Label }; if ($item.Label -eq 'pretrained checkpoint') {$PretrainedCheckpoint=$resolved} else {$ResumeFrom=$resolved} }
}
$outputDirectory=Join-Path $RunsRoot "finetune\$splitSlug\$Task\$FinetuneMode\budget-$LabelBudget\seed-$Seed"; Assert-OutputDirectoryAvailable -Path $outputDirectory -Force:$Force -DryRun:$DryRun
$arguments=@($entryPoint,"trainer=$Trainer","task=$Task","data.root=$CanonicalRoot","data.manifest_path=$SplitsPath","seed=$Seed","label_budget=$LabelBudget","finetune_mode=$FinetuneMode","unfrozen_blocks=$UnfrozenBlocks","hydra.run.dir=$outputDirectory")
if ($PretrainedCheckpoint) {$arguments += "pretrained_checkpoint=$PretrainedCheckpoint"}; if ($ResumeFrom) {$arguments += "resume_from=$ResumeFrom"}
Invoke-Checked $Python $arguments -DryRun:$DryRun
