# scripts/audit.ps1
# Tarkoitus: tarkista että Tutkija-projektin vaiheen 0 valmius ja yhteiset pelisäännöt ovat kunnossa.
# Suorita PowerShellissä projektin juuressa.

$ErrorActionPreference = "Stop"

function Ok($message)   { Write-Host "[OK]  $message" -ForegroundColor Green }
function Warn($message) { Write-Host "[WARN] $message" -ForegroundColor Yellow }
function Fail($message) { Write-Host "[FAIL] $message" -ForegroundColor Red }

$failures = @()
$warnings = @()

function Check-True($condition, $okMessage, $failMessage) {
  if ($condition) { Ok $okMessage }
  else { Fail $failMessage; $script:failures += $failMessage }
}

function Check-FileContains($path, $pattern, $description) {
  if (Test-Path $path) {
    $content = Get-Content -Raw -Encoding UTF8 $path
    if ($content -match $pattern) { Ok $description }
    else {
      $msg = "$description, ei löytynyt kaavaa: $pattern"
      Fail $msg
      $script:failures += $msg
    }
  } else {
    $msg = "$description, tiedosto puuttuu: $path"
    Fail $msg
    $script:failures += $msg
  }
}

$script:GhBinary = $null
try {
  $script:GhBinary = (Get-Command gh -ErrorAction Stop).Source
} catch {
  $fallbackGh = 'C:\\Program Files\\GitHub CLI\\gh.exe'
  if (Test-Path $fallbackGh) { $script:GhBinary = $fallbackGh }
}

function Has-Gh { return [bool]$script:GhBinary }

function Invoke-Gh {
  param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)
  if (-not (Has-Gh)) {
    throw 'gh CLI ei ole käytettävissä'
  }
  & $script:GhBinary @Arguments
}

Write-Host '== Tutkija audit =='

# 0. peruspolut ja työkalut
$branch = git rev-parse --abbrev-ref HEAD 2>$null
$onPr = [bool]$env:GITHUB_HEAD_REF
if (($branch -eq 'main') -or $onPr) {
  Ok 'oletushaara on main'
} else {
  Warn "ajo haarassa, $branch, paatestit kohdistuvat mainiin"
}
Check-True (Test-Path '.git') 'git repo löytyi' 'git repo puuttuu, aja git init'
Check-True (Test-Path '.venv/Scripts/Activate.ps1') '.venv löytyy' '.venv puuttuu, luo virtuaaliympäristö'

# 1. dokumentit
Check-True (Test-Path 'docs') 'docs-kansio on olemassa' 'docs-kansio puuttuu'
Check-True (Test-Path 'docs/ARCHITECTURE.md') 'arkkitehtuuri yhteenveto löytyy' 'docs/ARCHITECTURE.md puuttuu'
Check-True (Test-Path 'docs/pelisaannot.md') 'pelisäännöt dokumentti löytyy' 'docs/pelisaannot.md puuttuu'
Check-True (Test-Path 'docs/mittarit.md') 'mittarit dokumentti löytyy' 'docs/mittarit.md puuttuu'
Check-True (Test-Path 'docs/adr') 'ADR-kansio löytyy' 'docs/adr kansio puuttuu'
Check-True (Test-Path 'docs/adr/0001-tyokalupino.md') 'ADR 0001 löytyy' 'ADR 0001 puuttuu'

# 2. työtavat
Check-FileContains 'docs/pelisaannot.md' 'trunk|feature|PR|squash|semanttiset|Conventional' 'pelisäännöt kuvaavat trunk-based ja PR-käytännöt'
Check-FileContains 'docs/pelisaannot.md' 'Definition of Ready|DoR|Definition of Done|DoD' 'pelisäännöissä on DoR ja DoD kuvaukset'

# 3. CI ja paikalliset työkalut
Check-True (Test-Path '.github/workflows/ci.yml') 'CI workflow löytyy' 'CI workflow puuttuu (.github/workflows/ci.yml)'
Check-True (Test-Path '.pre-commit-config.yaml') 'pre-commit asetukset löytyvät' 'pre-commit asetukset puuttuvat'

$ruff = Get-Command ruff -ErrorAction SilentlyContinue
$mypy = Get-Command mypy -ErrorAction SilentlyContinue
$pytest = Get-Command pytest -ErrorAction SilentlyContinue
if (-not $ruff) { $script:warnings += 'ruff ei ole komentopolussa'; Warn 'ruff ei ole komentopolussa, aktivoi .venv' } else { Ok 'ruff löytyy' }
if (-not $mypy) { $script:warnings += 'mypy ei ole komentopolussa'; Warn 'mypy ei ole komentopolussa' } else { Ok 'mypy löytyy' }
if (-not $pytest) { $script:warnings += 'pytest ei ole komentopolussa'; Warn 'pytest ei ole komentopolussa' } else { Ok 'pytest löytyy' }

# 4. artefaktit gitin ulkopuolella
Check-True (Test-Path '.gitignore') '.gitignore löytyy' '.gitignore puuttuu'
if (Test-Path '.gitignore') {
  $gitignore = Get-Content -Raw -Encoding UTF8 '.gitignore'
  foreach ($dir in @('data/', 'cache/', 'output/')) {
    if ($gitignore -notmatch [regex]::Escape($dir)) {
      $msg = ".gitignore ei sisällä $dir"
      Fail $msg
      $script:failures += $msg
    } else {
      Ok ".gitignore sisältää $dir"
    }
  }
}

# 5. konfiguraatiot
Check-True (Test-Path '.env.example') '.env.example löytyy' '.env.example puuttuu'
Check-True (Test-Path 'config.example.toml') 'config.example.toml löytyy' 'config.example.toml puuttuu'

# 6. arkkitehtuuri
Check-FileContains 'docs/ARCHITECTURE.md' 'Komponentit|Dataflow' 'arkkitehtuuri sisältää komponentit ja dataflown'

# 7. CLI hello
Check-True (Test-Path 'src/la_pkg/cli.py') 'CLI-lähde löytyy' 'src/la_pkg/cli.py puuttuu'
try {
  $cliExe = Join-Path '.venv/Scripts' 'la.exe'
  if (Test-Path $cliExe) {
    $hello = & $cliExe hello 2>$null
  } else {
    $hello = la hello 2>$null
  }
  if ($LASTEXITCODE -eq 0 -and $hello -match 'Tutkija' -and $hello -match 'OPENAI_API_KEY') {
    Ok 'la hello toimii ja tulostaa .env.example mallin'
  } else {
    $msg = 'la hello ei tulostanut odotettua sisältöä'
    Fail $msg
    $script:failures += $msg
  }
} catch {
  $msg = "la hello ei käynnisty, $_"
  Fail $msg
  $script:failures += 'la hello crash'
}

# 7b. hakulogit ja yhdistetyt artefaktit
Check-True (Test-Path 'data/cache/search_log.csv') 'search_log.csv loytyi' 'data/cache/search_log.csv puuttuu, aja la search'

$mergedPath = 'data/cache/merged.parquet'
if (Test-Path $mergedPath) {
  Ok 'data/cache/merged.parquet loytyi'
} else {
  $msg = 'data/cache/merged.parquet puuttuu, aja la search-all'
  Fail $msg
  $script:failures += $msg
}



$mergeLogPath = 'data/cache/merge_log.csv'
$expectedMergeCols = @('topic', 'time', 'per_source_counts', 'duplicates_by_doi', 'duplicates_by_title', 'filtered_by_rules', 'final_count', 'out_path')
if (Test-Path $mergeLogPath) {
  $rows = @()
  try {
    $rows = Import-Csv -Path $mergeLogPath
  } catch {
    $rows = @()
  }
  if ($rows.Count -gt 0) {
    $first = $rows[0]
    $headers = $first.PSObject.Properties.Name
    $missing = @($expectedMergeCols | Where-Object { $headers -notcontains $_ })
    if ($missing.Count -eq 0) {
      Ok 'merge_log.csv sarakkeet kunnossa'
    } else {
      $msg = "merge_log.csv puuttuu sarakkeet: $($missing -join ', ')"
      Fail $msg
      $script:failures += $msg
    }
    $last = $rows[-1]
    $counts = $null
    try {
      $counts = $last.per_source_counts | ConvertFrom-Json
    } catch {
      $counts = $null
    }
    if ($counts) {
      $countKeys = $counts.PSObject.Properties.Name
      $requiredSources = @('openalex', 'pubmed', 'arxiv')
      $missingSources = @($requiredSources | Where-Object { $countKeys -notcontains $_ })
      if ($missingSources.Count -eq 0) {
        $hasLiveData = $false
        foreach ($source in $requiredSources) {
          $value = $counts.$source
          if ($value -and [int]$value -gt 0) { $hasLiveData = $true; break }
        }
        if ($hasLiveData) {
          Ok 'merge_log per_source_counts l?ytyi ja sis?lt?? live dataa'
        } else {
          $msg = 'per_source_counts kaikki arvot 0, todenn?k?isesti ei live hakua'
          Fail $msg
          $script:failures += $msg
        }
      } else {
        $msg = "per_source_counts puuttuu avaimet: $($missingSources -join ', ')"
        Fail $msg
        $script:failures += $msg
      }
    } else {
      $msg = 'per_source_counts ei ole kelvollista JSONia'
      Fail $msg
      $script:failures += $msg
    }
    $outPath = $last.out_path
    if ($outPath -and (Test-Path $outPath)) {
      Ok "merge_log out_path viittaa olemassa olevaan tiedostoon: $outPath"
    } else {
      $msg = "merge_log out_path puuttuu tai tiedosto ei ole olemassa: $outPath"
      Fail $msg
      $script:failures += $msg
    }
  } else {
    Warn 'merge_log.csv on tyhja, aja la search-all'
    $script:warnings += 'merge log empty'
  }
} else {
  $msg = 'data/cache/merge_log.csv puuttuu, aja la search-all'
  Fail $msg
  $script:failures += $msg
}
# 8. viimeisin CI-ajo
if (Has-Gh) {
  try {
    $runJson = Invoke-Gh run list --limit 1 --json status --json conclusion --json name --jq '.[0]'
    if ($runJson) {
      $run = $runJson | ConvertFrom-Json
      if ($run.status -eq 'completed' -and $run.conclusion -eq 'success') {
        Ok 'viimeisin GitHub Actions -ajo on vihreä'
      } else {
        $msg = "viimeisin GitHub Actions -ajo ei ole vihreä, status=$($run.status), conclusion=$($run.conclusion)"
        Warn $msg
        $script:warnings += 'CI not green'
      }
    } else {
      Warn 'ei löytynyt Actions-ajoja, puske main tai avaa PR'
      $script:warnings += 'no CI runs'
    }
  } catch {
    Warn 'gh run list epäonnistui, ohitetaan CI-tarkistus'
    $script:warnings += 'gh error'
  }
} else {
  Warn 'gh CLI ei ole käytettävissä, CI-tarkistus ohitetaan'
  $script:warnings += 'no gh'
}

# 9. README
if (Test-Path 'README.md') {
  $readme = Get-Content -Raw -Encoding UTF8 'README.md'
  if ($readme -match 'Asennus|Installation|setup|make setup|uv venv') { Ok 'README sisältää asennusohjeen' }
  else { Warn 'README ei kuvaa asennusta'; $script:warnings += 'readme install' }
  if ($readme -match 'la hello') { Ok 'README sisältää nopean testin' }
  else { Warn 'README ei sisällä nopeaa testiä'; $script:warnings += 'readme quicktest' }
} else {
  $msg = 'README.md puuttuu'
  Fail $msg
  $script:failures += $msg
}

# 10. branch protection
if (Has-Gh) {
  try {
    $originUrl = git remote get-url origin 2>$null
    if ($originUrl) {
      $ownerRepo = $null
      if ($originUrl -match 'github.com[:/](.+?)(?:\.git)?$') {
        $ownerRepo = $matches[1]
        if ($ownerRepo -and $ownerRepo.EndsWith('.git')) {
          $ownerRepo = $ownerRepo.Substring(0, $ownerRepo.Length - 4)
        }
      }
      if ($ownerRepo) {
        $response = Invoke-Gh api "repos/$ownerRepo/branches/main/protection" -H 'Accept: application/vnd.github+json'
        if ($response) {
          $prot = $response | ConvertFrom-Json
          Ok 'branch protection on asetettu main haaralle'
          $contexts = $prot.required_status_checks.contexts
          if ($contexts -and $contexts.Count -gt 0) {
            Ok "required status checks: $($contexts -join ', ')"
          } else {
            Warn 'required status checks puuttuu, lisää build ja audit'
            $script:warnings += 'no required checks'
          }
          if ($prot.required_conversation_resolution.enabled) {
            Ok 'keskustelujen ratkaisu vaaditaan ennen mergeä'
          }
        } else {
          Warn 'branch protection tietoja ei saatu haettua'
          $script:warnings += 'bp fetch empty'
        }
      } else {
        Warn 'origin remote osoitetta ei tunnistettu, ohitetaan suojauksen tarkistus'
        $script:warnings += 'no origin'
      }
    } else {
      Warn 'origin remote puuttuu, ohitetaan suojauksen tarkistus'
      $script:warnings += 'no origin'
    }
  } catch {
    Warn "branch protection tarkistus epäonnistui, $_"
    $script:warnings += 'bp error'
  }
} else {
  Warn 'gh CLI ei ole käytettävissä, branch protection tarkistus ohitetaan'
  $script:warnings += 'no gh'
}

Write-Host ''
if ($failures.Count -eq 0) {
  Ok 'Audit, pakolliset kohdat kunnossa'
} else {
  Fail "Audit, pakollisia puutteita: $($failures.Count) kpl"
}
if ($warnings.Count -gt 0) {
  Warn "Huomioita: $($warnings.Count), suositellaan korjattavaksi"
}

if ($failures.Count -gt 0) { exit 1 } else { exit 0 }
