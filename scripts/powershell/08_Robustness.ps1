[CmdletBinding()]
param(
    [string]$ProjectRoot = 'D:\Fmodel\metafmodel', [string]$CanonicalRoot = 'D:\Fmodel\emg_data_canonical',
    [string]$RunsRoot = 'D:\Fmodel\runs', [string]$SplitsPath = '',
    [ValidateSet('pose','wrist_angle','discrete_gesture','handwriting','typing')][string]$Task = 'pose',
    [ValidateSet('All','none','channel_rotation','channel_dropout','gain_drift','additive_noise','temporal_loss')][string]$Condition = 'All',
    [Parameter(Mandatory=$true)][string]$CheckpointPath, [int]$Seed = 0, [switch]$Smoke, [string]$Python = 'python', [switch]$DryRun
)
Set-StrictMode -Version Latest; $ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'Common.ps1')
$ProjectRoot=Resolve-FullPath $ProjectRoot; $CanonicalRoot=Resolve-FullPath $CanonicalRoot; $RunsRoot=Resolve-FullPath $RunsRoot
if ([string]::IsNullOrWhiteSpace($SplitsPath)) {$SplitsPath=Join-Path $CanonicalRoot 'splits.csv'} else {$SplitsPath=Resolve-FullPath $SplitsPath}
$CheckpointPath=Resolve-FullPath $CheckpointPath; $entryPoint=Join-Path $ProjectRoot 'evaluate.py'
Assert-File $entryPoint 'evaluate.py'; Assert-Directory $CanonicalRoot 'canonical data root'; Assert-File $SplitsPath 'split manifest'; if(-not $DryRun){Assert-File $CheckpointPath 'trusted repository checkpoint'}
$conditions=if($Condition -eq 'All'){@('none','channel_rotation','channel_dropout','gain_drift','additive_noise','temporal_loss')}else{@($Condition)}; $trainer=if($Smoke){'cpu_smoke'}else{'a100_ddp'}
foreach($item in $conditions){$outputDirectory=Join-Path $RunsRoot "robustness\$Task\$item"; $arguments=@($entryPoint,"trainer=$trainer","task=$Task","robustness=$item","data.root=$CanonicalRoot","data.manifest_path=$SplitsPath","checkpoint_path=$CheckpointPath","output_dir=$outputDirectory","seed=$Seed","hydra.run.dir=$outputDirectory\hydra"); Invoke-Checked $Python $arguments -DryRun:$DryRun}
