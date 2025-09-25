# scripts/audit.ps1
# Tarkoitus, tarkista että Tutkija projektissa on tehty yhteiset pelisäännöt ja vaihe 0
# Aja PowerShellissä projektin juuressa

$ErrorActionPreference = "Stop"

function Ok($m)   { Write-Host "[OK]  $m" -ForegroundColor Green }
function Warn($m) { Write-Host "[WARN] $m" -ForegroundColor Yellow }
function Fail($m) { Write-Host "[FAIL] $m" -ForegroundColor Red }

$fail = @()
$warn = @()

function Check-True($cond, $ok, $failmsg) {
  if ($cond) { Ok $ok }
  else { Fail $failmsg; $script:fail += $failmsg }
}

function Check-FileContains($path, $pattern, $desc) {
  if (Test-Path $path) {
    $txt = Get-Content -Raw -Encoding UTF8 $path
    if ($txt -match $pattern) { Ok $desc }
    else {
      Fail "$desc, ei läpäissyt haun, $pattern"
      $script:fail += "$desc pattern $pattern"
    }
  } else {
    Fail "$desc, tiedosto puuttuu, $path"
    $script:fail += "$desc missing"
  }
}

$script:GhCommand = $null
try {
  $script:GhCommand = (Get-Command gh -ErrorAction Stop).Source
} catch {
  $defaultGh = 'C:\\Program Files\\GitHub CLI\\gh.exe'
  if (Test-Path $defaultGh) {
    $script:GhCommand = $defaultGh
  }
}

function Has-Gh {
  return [bool]$script:GhCommand
}

function Invoke-Gh {
  param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Args)
  if (-not (Has-Gh)) {
    throw 'gh CLI ei ole käytettävissä'
  }
  & $script:GhCommand @Args
}

Write-Host "== Tutkija audit =="

# 0, peruspolut ja työkalut
Check-True (Test-Path ".git") "git repo löytyi" "git repo puuttuu, aja git init"
Check-True ((git rev-parse --abbrev-ref HEAD 2>$null) -eq "main") "oletushaara on main" "oletushaara ei ole main"
Check-True (Test-Path ".venv/Scripts/Activate.ps1") ".venv löytyy" ".venv puuttuu, luo virtuaaliympäristö"

# 1, yhteiset pelisäännöt, yksi totuuden lähde
Check-True (Test-Path "docs") "docs kansio on olemassa" "docs kansio puuttuu"
Check-True (Test-Path "docs/ARCHITECTURE.md") "arkkitehtuuri yhteenveto löytyy" "docs/ARCHITECTURE.md puuttuu"
Check-True (Test-Path "docs/pelisaannot.md") "pelisäännöt dokumentti löytyy" "docs/pelisaannot.md puuttuu"
Check-True (Test-Path "docs/mittarit.md") "mittarit dokumentti löytyy" "docs/mittarit.md puuttuu"
Check-True (Test-Path "docs/adr") "ADR kansio löytyy" "docs/adr kansio puuttuu"
Check-True (Test-Path "docs/adr/0001-tyokalupino.md") "ADR 0001, työkalupino löytyy" "ADR 0001 puuttuu"

# 2, työtavat
Check-FileContains "docs/pelisaannot.md" "trunk|feature|PR|squash|semanttiset|Conventional" "pelisäännöt kuvaavat trunk based ja PR käytännöt"
Check-FileContains "docs/pelisaannot.md" "Definition of Ready|DoR|Definition of Done|DoD" "pelisäännöissä on DoR ja DoD kuvaukset"

# 3, CI, lint ja testit
Check-True (Test-Path ".github/workflows/ci.yml") "CI workflow löytyy" "CI workflow puuttuu, .github/workflows/ci.yml"
Check-True (Test-Path ".pre-commit-config.yaml") "pre-commit asetukset löytyvät" "pre-commit asetukset puuttuvat"
$ruff = Get-Command ruff -ErrorAction SilentlyContinue
$mypy = Get-Command mypy -ErrorAction SilentlyContinue
$pytest = Get-Command pytest -ErrorAction SilentlyContinue
if (-not $ruff) { $script:warn += "ruff ei ole komentopolussa, varmistetaan requirements ja pyproject"; Warn "ruff ei ole komentopolussa, varmistetaan requirements ja pyproject" } else { Ok "ruff löytyy" }
if (-not $mypy) { $script:warn += "mypy ei ole komentopolussa"; Warn "mypy ei ole komentopolussa" } else { Ok "mypy löytyy" }
if (-not $pytest) { $script:warn += "pytest ei ole komentopolussa"; Warn "pytest ei ole komentopolussa" } else { Ok "pytest löytyy" }

# 4, artefaktit pois gitistä
Check-True (Test-Path ".gitignore") ".gitignore löytyy" ".gitignore puuttuu"
if (Test-Path ".gitignore") {
  $gi = Get-Content -Raw -Encoding UTF8 .gitignore
  foreach ($must in @("data/", "cache/", "output/")) {
    if ($gi -notmatch [regex]::Escape($must)) {
      Fail ".gitignore ei sisällä $must"
      $script:fail += ".gitignore missing $must"
    } else {
      Ok ".gitignore sisältää $must"
    }
  }
}

# 5, peruskonffit ja templatet
Check-True (Test-Path ".env.example") ".env.example löytyy" ".env.example puuttuu"
Check-True (Test-Path "config.example.toml") "config.example.toml löytyy" "config.example.toml puuttuu"

# 6, lyhyt arkkitehtuuri ja dataflow
Check-FileContains "docs/ARCHITECTURE.md" "Komponentit|Dataflow" "arkkitehtuuri, sisältää komponentti ja dataflow osiot"

# 7, CLI hello
Check-True (Test-Path "src/la_pkg/cli.py") "CLI lähde löytyy" "src/la_pkg/cli.py puuttuu"
try {
  $laPath = Join-Path ".venv/Scripts" "la.exe"
  if (Test-Path $laPath) {
    $hello = & $laPath hello 2>$null
  } else {
    $hello = la hello 2>$null
  }
  if ($LASTEXITCODE -eq 0 -and $hello -match "Tutkija" -and $hello -match "OPENAI_API_KEY") {
    Ok "la hello toimii ja tulostaa .env.example mallin"
  } else {
    Fail "la hello ei tulostanut odotettua sisältöä"
    $script:fail += "la hello output"
  }
} catch {
  Fail "la hello ei käynnisty, $_"
  $script:fail += "la hello crash"
}

# 8, CI vihreänä
if (Has-Gh) {
  try {
    $runsJson = Invoke-Gh run list --limit 1 --json status,conclusion,name -q '.[0]'
    if ($runsJson) {
      $obj = $runsJson | ConvertFrom-Json
      if ($obj.status -eq "completed" -and $obj.conclusion -eq "success") {
        Ok "viimeisin GitHub Actions ajo on vihreä"
      } else {
        Warn "viimeisin GitHub Actions ajo ei ole vihreä, status, $($obj.status), conclusion, $($obj.conclusion)"
        $script:warn += "CI not green"
      }
    } else {
      Warn "ei löytynyt Actions ajoja, puske main tai avaa PR"
      $script:warn += "no CI runs"
    }
  } catch {
    Warn "gh run list epäonnistui, ohitetaan CI tarkistus"
    $script:warn += "gh error"
  }
} else {
  Warn "gh CLI ei ole käytettävissä, CI tarkistus ohitetaan"
  $script:warn += "no gh"
}

# 9, README, asennus ja testi
if (Test-Path "README.md") {
  $rd = Get-Content -Raw -Encoding UTF8 README.md
  if ($rd -match "Asennus|Installation|setup|make setup|uv venv") { Ok "README sisältää asennusohjeen" }
  else { Warn "README ei kuvaa asennusta, lisää pikaohje"; $script:warn += "readme install" }
  if ($rd -match "la hello") { Ok "README sisältää nopean testin" }
  else { Warn "README ei sisällä nopeaa testiä, lisää la hello esimerkki"; $script:warn += "readme quicktest" }
} else {
  Fail "README.md puuttuu"
  $script:fail += "readme missing"
}

# 10, Exit kriteerit
if (Has-Gh) {
  try {
    $originUrl = git remote get-url origin 2>$null
    if ($originUrl) {
      $ownerRepo = $originUrl -replace '.*github.com[:/](.*)\.git','$1'
      if ($ownerRepo) {
        $prot = Invoke-Gh api repos/$ownerRepo/branches/main/protection -H "Accept: application/vnd.github+json" 2>$null
        if ($prot) {
          Ok "branch protection on asetettu main haaralle"
          $ctx = ($prot | ConvertFrom-Json).required_status_checks.contexts
          if ($ctx -and $ctx.Count -gt 0) {
            Ok "required status checks on määritelty, $($ctx -join ', ')"
          } else {
            Warn "required status checks puuttuu, lisää build, ruff, mypy, pytest"
            $script:warn += "no required checks"
          }
        } else {
          Warn "branch protection ei palauttanut tietoja, varmista oikeudet tai aseta suojaus"
          $script:warn += "no branch protection"
        }
      } else {
        Warn "origin remote osoitetta ei tunnistettu, ohitetaan suojauksen tarkistus"
        $script:warn += "no origin"
      }
    } else {
      Warn "origin remote puuttuu, ohitetaan suojauksen tarkistus"
      $script:warn += "no origin"
    }
  } catch {
    Warn "branch protection tarkistus epäonnistui, $_"
    $script:warn += "bp error"
  }
} else {
  Warn "gh CLI ei ole käytettävissä, branch protection tarkistus ohitetaan"
  $script:warn += "no gh"
}

Write-Host ""
if ($fail.Count -eq 0) { Ok "Audit, pakolliset kohdat kunnossa" }
else { Fail "Audit, pakollisia puutteita, $($fail.Count) kohtaa" }
if ($warn.Count -gt 0) { Warn "Huomioita, $($warn.Count), nämä eivät estä etenemistä mutta suositellaan korjattavaksi" }

if ($fail.Count -gt 0) { exit 1 } else { exit 0 }
