[CmdletBinding()]
param(
    [string]$ProjectRoot = 'D:\Fmodel\metafmodel',
    [string]$CanonicalRoot = 'D:\Fmodel\emg_data_canonical',
    [string]$ManifestPath = '',
    [ValidateSet('Official', 'StrictSubject')][string]$Protocol = 'Official',
    [int]$Seed = 20260916,
    [string]$Python = 'python',
    [switch]$Force,
    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'Common.ps1')

$ProjectRoot = Resolve-FullPath -Path $ProjectRoot
$CanonicalRoot = Resolve-FullPath -Path $CanonicalRoot
if ([string]::IsNullOrWhiteSpace($ManifestPath)) {
    $ManifestPath = Join-Path $CanonicalRoot 'manifest.csv'
}
else {
    $ManifestPath = Resolve-FullPath -Path $ManifestPath
}

$splitEntry = Join-Path $ProjectRoot 'make_splits.py'
Assert-Directory -Path $CanonicalRoot -Label 'canonical data root'
Assert-File -Path $ManifestPath -Label 'source manifest'
Assert-File -Path $splitEntry -Label 'make_splits.py'

$effectiveManifest = $ManifestPath
if ($Protocol -eq 'StrictSubject') {
    $effectiveManifest = Join-Path $CanonicalRoot 'manifest.strict-subject.csv'
}
$protocolSlug = if ($Protocol -eq 'Official') { 'official' } else { 'strict-subject' }
$outputPath = Join-Path $CanonicalRoot "splits.$protocolSlug.seed-$Seed.csv"

foreach ($candidate in @($effectiveManifest, $outputPath)) {
    if ($candidate -eq $ManifestPath) {
        continue
    }
    if ((Test-Path -LiteralPath $candidate) -and -not $Force) {
        throw "output already exists; pass -Force to replace it: $candidate"
    }
}

if ($Protocol -eq 'StrictSubject') {
    Write-Host "Strict-subject derived manifest: $effectiveManifest"
    if (-not $DryRun) {
        $rows = Import-Csv -LiteralPath $ManifestPath
        if ($rows.Count -eq 0 -or -not ($rows[0].PSObject.Properties.Name -contains 'official_split')) {
            throw "manifest must contain official_split: $ManifestPath"
        }
        foreach ($row in $rows) {
            $row.official_split = 'unknown'
        }
        $rows | Export-Csv -LiteralPath $effectiveManifest -NoTypeInformation -Encoding UTF8 -Force
    }
}
else {
    Write-Host "Official manifest: $effectiveManifest"
}
Write-Host "Split output: $outputPath"

$arguments = @(
    $splitEntry,
    '--manifest', $effectiveManifest,
    '--root', $CanonicalRoot,
    '--output', $outputPath,
    '--seed', [string]$Seed
)
Invoke-Checked -Executable $Python -Arguments $arguments -DryRun:$DryRun

if (-not $DryRun) {
    Assert-File -Path $outputPath -Label 'generated split manifest'
    $assigned = Import-Csv -LiteralPath $outputPath
    Write-Host 'Task-by-split counts:'
    $assigned |
        Group-Object task_id, split |
        Sort-Object Name |
        ForEach-Object { Write-Host ("{0}: {1}" -f $_.Name, $_.Count) }

    if ($Protocol -eq 'StrictSubject') {
        $subjectLeaks = @(
            $assigned |
                Group-Object { "$($_.dataset_id)|$($_.subject_id)" } |
                Where-Object { @($_.Group.split | Select-Object -Unique).Count -gt 1 }
        )
        $paired = @($assigned | Where-Object { -not [string]::IsNullOrWhiteSpace($_.pair_id) })
        $pairLeaks = @(
            $paired |
                Group-Object { "$($_.dataset_id)|$($_.pair_id)" } |
                Where-Object { @($_.Group.split | Select-Object -Unique).Count -gt 1 }
        )
        if ($subjectLeaks.Count -gt 0 -or $pairLeaks.Count -gt 0) {
            throw "strict-subject leakage detected: subjects=$($subjectLeaks.Count), pairs=$($pairLeaks.Count)"
        }
        Write-Host 'Strict-subject leakage check: PASS'
    }
    else {
        Write-Host 'Official release assignments preserved; release-defined cross-subject overlap is reported by make_splits.py.'
    }
}
