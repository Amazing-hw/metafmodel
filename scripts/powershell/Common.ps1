Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'


function Resolve-FullPath {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [string]$BasePath = (Get-Location).Path
    )

    if ([IO.Path]::IsPathRooted($Path)) {
        return [IO.Path]::GetFullPath($Path)
    }
    return [IO.Path]::GetFullPath((Join-Path $BasePath $Path))
}


function Assert-File {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Label
    )

    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "$Label not found: $Path"
    }
}


function Assert-Directory {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Label
    )

    if (-not (Test-Path -LiteralPath $Path -PathType Container)) {
        throw "$Label not found: $Path"
    }
}


function Format-Command {
    param(
        [Parameter(Mandatory = $true)][string]$Executable,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )

    $quoted = $Arguments | ForEach-Object {
        if ($_ -match '[\s"]') {
            '"' + ($_ -replace '"', '\"') + '"'
        }
        else {
            $_
        }
    }
    return (@($Executable) + @($quoted)) -join ' '
}


function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)][string]$Executable,
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [switch]$DryRun
    )

    Write-Host (Format-Command -Executable $Executable -Arguments $Arguments)
    if ($DryRun) {
        return
    }
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Executable exited with code $LASTEXITCODE"
    }
}


function Ensure-OutputDirectory {
    param([Parameter(Mandatory = $true)][string]$Path)

    if (-not (Test-Path -LiteralPath $Path)) {
        New-Item -ItemType Directory -Path $Path | Out-Null
    }
    return (Resolve-Path -LiteralPath $Path).Path
}

function Resolve-SplitManifest {
    param(
        [Parameter(Mandatory = $true)][string]$CanonicalRoot,
        [string]$SplitsPath = '',
        [string]$Protocol = '',
        [Nullable[int]]$SplitSeed = $null,
        [switch]$AllowMissing
    )

    if (-not [string]::IsNullOrWhiteSpace($SplitsPath)) {
        $resolved = Resolve-FullPath -Path $SplitsPath
        if (-not $AllowMissing) {
            Assert-File -Path $resolved -Label 'selected split manifest'
        }
        return $resolved
    }
    if ([string]::IsNullOrWhiteSpace($Protocol) -or $null -eq $SplitSeed) {
        throw 'split selection requires -SplitsPath or both -Protocol and -SplitSeed'
    }
    $slug = if ($Protocol -eq 'Official') { 'official' } else { 'strict-subject' }
    $candidate = Join-Path $CanonicalRoot "splits.$slug.seed-$SplitSeed.csv"
    if (-not $AllowMissing) {
        Assert-File -Path $candidate -Label 'selected split manifest'
    }
    return (Resolve-FullPath -Path $candidate)
}

function Get-SplitRunSlug {
    param([string]$SplitsPath, [string]$Protocol = '', [Nullable[int]]$SplitSeed = $null)

    if (-not [string]::IsNullOrWhiteSpace($Protocol) -and $null -ne $SplitSeed) {
        $slug = if ($Protocol -eq 'Official') { 'official' } else { 'strict-subject' }
        return "$slug-seed-$SplitSeed"
    }
    $bytes = [Text.Encoding]::UTF8.GetBytes($SplitsPath)
    $hash = [Security.Cryptography.SHA256]::Create().ComputeHash($bytes)
    $hex = ([BitConverter]::ToString($hash) -replace '-', '').ToLowerInvariant()
    return "custom-$($hex.Substring(0, 12))"
}

function Assert-OutputDirectoryAvailable {
    param([string]$Path, [switch]$Force, [switch]$DryRun)

    if ((Test-Path -LiteralPath $Path) -and -not $Force) {
        throw "output directory already exists; pass -Force to reuse it: $Path"
    }
    if (-not $DryRun -and -not (Test-Path -LiteralPath $Path)) {
        New-Item -ItemType Directory -Path $Path | Out-Null
    }
}
