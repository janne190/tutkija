[![CI](https://github.com/janne190/tutkija/actions/workflows/ci.yml/badge.svg)](https://github.com/janne190/tutkija/actions/workflows/ci.yml)
# Tutkija

Tutkija on projektipohja moniagenttiselle kirjallisuuskatsaukselle. Repossa mÃ¤Ã¤ritellÃ¤Ã¤n yhteiset pelisÃ¤Ã¤nnÃ¶t, kehitystyÃ¶kalut ja vaiheittainen etenemissuunnitelma ennen varsinaista toteutusta.

## Dokumentaatio
- `docs/pelisaannot.md` â€“ tiimin yhteiset kÃ¤ytÃ¤nnÃ¶t ja DoR/DoD
- `docs/ARCHITECTURE.md` â€“ arkkitehtuurin yksi sivu
- `docs/mittarit.md` â€“ valvottavat mittarit ja kynnysarvot
- `docs/adr/` â€“ arkkitehtuuripÃ¤Ã¤tÃ¶sten loki

## Kehitysvaiheet
- Vaihe 0: perusta ja CLI-rungon savutesti (`la hello`)
- Vaihe 1: haku + metatieto tuotantokunnolla, golden snapshot ja mittarit
- Vaihe 2: laajennettu seulonta, PDF-jÃ¤sentÃ¤minen ja raportointi automatisoituna

Aja `make setup` (tai toteuta vastaavat komennot kÃ¤sin) ennen ensimmÃ¤istÃ¤ muutosta ja varmista pre-commit-koukut komennolla `pre-commit run --all-files`.

## Julkaisu
Tagaa versio muodossa vX.Y.Z, esimerkki, v0.0.1. CI rakentaa paketit ja tekee GitHub releasen automaattisesti.
