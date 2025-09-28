# tools/run_vaihe4.ps1
# Purpose: orchestrate the Tutkija Phase 4 workflow (PDF discovery, download, parsing).
# Run this from the project root in Windows PowerShell.

[CmdletBinding()]
param(
  [switch]$SkipDiscover,
  [string]$MetadataPath = "data/cache/merged.parquet",
  [string]$SeedCsv = "tools/seed_urls.csv",
  [string]$PdfIndex = "data/cache/pdf_index.parquet",
  [string]$PdfDir = "data/pdfs",
  [string]$AuditLog = "data/logs/pdf_audit.csv",
  [string]$ParsedDir = "data/parsed",
  [string]$ParsedIndex = "data/cache/parsed.parquet",
  [string]$ParseErrors = "data/logs/parse_errors.csv",
  [string]$GrobidUrl
)

$ErrorActionPreference = "Stop"

function Write-Step([string]$Message)   { Write-Host "== $Message" -ForegroundColor Cyan }
function Write-Ok([string]$Message)     { Write-Host "[OK]  $Message" -ForegroundColor Green }
function Write-Warn([string]$Message)   { Write-Host "[WARN] $Message" -ForegroundColor Yellow }
function Write-Fail([string]$Message)   { Write-Host "[FAIL] $Message" -ForegroundColor Red }

Write-Step "Tutkija Phase 4 runner"

if (-not (Test-Path .git)) {
  Write-Fail "Script must be executed from the repository root."
  exit 1
}

$branch = git rev-parse --abbrev-ref HEAD
$commit = git rev-parse --short HEAD
$dirty = git status --short
Write-Ok "Git branch: $branch"
Write-Ok "Git commit: $commit"
if ($dirty) {
  Write-Warn "Working tree has uncommitted changes."
}

Write-Step "Bootstrap virtual environment"
$venvActivate = Join-Path '.venv' 'Scripts/Activate.ps1'
if (-not (Test-Path $venvActivate)) {
  Write-Warn ".venv missing, creating a new virtual environment."
  python -m venv .venv
}

. $venvActivate

$uv = Get-Command uv -ErrorAction SilentlyContinue
if ($uv) {
  Write-Ok "uv detected: $($uv.Source)"
  uv pip install -e ".[parse]" | Out-Host
} else {
  Write-Warn "uv not found, falling back to pip."
  python -m pip install --upgrade pip | Out-Host
  python -m pip install -e ".[parse]" | Out-Host
}

$laExe = Join-Path '.venv/Scripts' 'la.exe'
if (-not (Test-Path $laExe)) {
  $laExe = 'la'
}

function Invoke-La {
  param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Args)
  & $laExe @Args
  if ($LASTEXITCODE -ne 0) {
    throw "la command failed: $($Args -join ' ')"
  }
}

Write-Step "Environment checks"
if ([string]::IsNullOrWhiteSpace($env:UNPAYWALL_EMAIL)) {
  Write-Warn "UNPAYWALL_EMAIL is not set; Unpaywall requests may fail."
} else {
  Write-Ok "UNPAYWALL_EMAIL present."
}

if (-not $PSBoundParameters.ContainsKey('GrobidUrl') -or [string]::IsNullOrWhiteSpace($GrobidUrl)) {
  if (-not [string]::IsNullOrWhiteSpace($env:GROBID_URL)) {
    $GrobidUrl = $env:GROBID_URL
    Write-Ok "Using GROBID_URL from environment: $GrobidUrl"
  } else {
    $GrobidUrl = 'http://localhost:8070'
    Write-Warn "GROBID_URL not set; defaulting to $GrobidUrl"
  }
} else {
  Write-Ok "Using GROBID_URL from parameter: $GrobidUrl"
}

New-Item -ItemType Directory -Force -Path (Split-Path $PdfIndex) | Out-Null
New-Item -ItemType Directory -Force -Path $PdfDir | Out-Null
New-Item -ItemType Directory -Force -Path (Split-Path $AuditLog) | Out-Null
New-Item -ItemType Directory -Force -Path $ParsedDir | Out-Null
New-Item -ItemType Directory -Force -Path (Split-Path $ParsedIndex) | Out-Null
New-Item -ItemType Directory -Force -Path (Split-Path $ParseErrors) | Out-Null

if (-not $SkipDiscover) {
  Write-Step "PDF discovery"
  if (-not (Test-Path $MetadataPath) -and -not (Test-Path $SeedCsv)) {
    Write-Warn "Neither metadata parquet ($MetadataPath) nor seed CSV ($SeedCsv) found; skipping discovery."
  } else {
    $discoverArgs = @('pdf', 'discover', '--out', $PdfIndex, '--seed-csv', $SeedCsv, '--in', $MetadataPath)
    Invoke-La @discoverArgs
    Write-Ok "PDF index written to $PdfIndex"
  }
} else {
  Write-Warn "SkipDiscover flag supplied; expecting $PdfIndex to exist."
}

if (-not (Test-Path $PdfIndex)) {
  Write-Fail "PDF index not found at $PdfIndex"
  exit 1
}

Write-Step "PDF download"
$mailto = if ([string]::IsNullOrWhiteSpace($env:UNPAYWALL_EMAIL)) { '' } else { $env:UNPAYWALL_EMAIL }
$downloadArgs = @('pdf', 'download', '--in', $PdfIndex, '--pdf-dir', $PdfDir, '--audit', $AuditLog)
if ($mailto) { $downloadArgs += @('--mailto', $mailto) }
Invoke-La @downloadArgs
Write-Ok "PDF download completed"

Write-Step "Parse PDFs"
$parseArgs = @('parse', 'run', '--pdf-dir', $PdfDir, '--out-dir', $ParsedDir, '--index-out', $ParsedIndex, '--grobid-url', $GrobidUrl, '--err-log', $ParseErrors)
Invoke-La @parseArgs
Write-Ok "Parse finished"

Write-Step "Summary"
$pdfCount = 0
$teiCount = 0
if (Test-Path $PdfDir) {
  $pdfCount = (Get-ChildItem -Path $PdfDir -Filter '*.pdf' -Recurse -ErrorAction SilentlyContinue | Measure-Object).Count
}
if (Test-Path $ParsedDir) {
  $teiCount = (Get-ChildItem -Path $ParsedDir -Filter 'tei.xml' -Recurse -ErrorAction SilentlyContinue | Measure-Object).Count
}
Write-Host ("PDF files: {0}" -f $pdfCount)
Write-Host ("Parsed TEI: {0}" -f $teiCount)
Write-Host "Index: $PdfIndex"
Write-Host "Parsed index: $ParsedIndex"
Write-Host "Parse errors: $ParseErrors"

Write-Step "Phase 4 workflow complete"
