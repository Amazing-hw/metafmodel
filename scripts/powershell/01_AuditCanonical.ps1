[CmdletBinding()]
param(
    [string]$ProjectRoot = 'D:\Fmodel\metafmodel',
    [string]$CanonicalRoot = 'D:\Fmodel\emg_data_canonical',
    [string]$RunsRoot = 'D:\Fmodel\runs',
    [string]$ManifestPath = '',
    [ValidateRange(0, 2147483647)][int]$H5Samples = 20,
    [string]$Python = 'python',
    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'Common.ps1')

$ProjectRoot = Resolve-FullPath -Path $ProjectRoot
$CanonicalRoot = Resolve-FullPath -Path $CanonicalRoot
$RunsRoot = Resolve-FullPath -Path $RunsRoot
if ([string]::IsNullOrWhiteSpace($ManifestPath)) {
    $ManifestPath = Join-Path $CanonicalRoot 'manifest.csv'
}
else {
    $ManifestPath = Resolve-FullPath -Path $ManifestPath
}

$auditEntry = Join-Path $ProjectRoot 'audit_canonical.py'
Assert-Directory -Path $CanonicalRoot -Label 'canonical data root'
Assert-File -Path $ManifestPath -Label 'canonical manifest'
Assert-File -Path $auditEntry -Label 'audit_canonical.py'

$auditDirectory = Join-Path $RunsRoot 'audits'
if (-not $DryRun) {
    $auditDirectory = Ensure-OutputDirectory -Path $auditDirectory
}
$timestamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$outputStem = Join-Path $auditDirectory "audit-$timestamp"
$arguments = @(
    $auditEntry,
    '--root', $CanonicalRoot,
    '--manifest', $ManifestPath,
    '--output-stem', $outputStem,
    '--h5-samples', [string]$H5Samples
)
Invoke-Checked -Executable $Python -Arguments $arguments -DryRun:$DryRun
