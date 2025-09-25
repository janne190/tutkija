# Tutkija

Tutkija on projektipohja moniagenttiselle kirjallisuuskatsaukselle. Repossa määritellään yhteiset pelisäännöt, kehitystyökalut ja vaiheittainen etenemissuunnitelma ennen varsinaista toteutusta.

## Dokumentaatio
- `docs/pelisaannot.md` – tiimin yhteiset käytännöt ja DoR/DoD
- `docs/ARCHITECTURE.md` – arkkitehtuurin yksi sivu
- `docs/mittarit.md` – valvottavat mittarit ja kynnysarvot
- `docs/adr/` – arkkitehtuuripäätösten loki

## Kehitysvaiheet
- Vaihe 0: perusta ja CLI-rungon savutesti (`la hello`)
- Vaihe 1: haku + metatieto tuotantokunnolla, golden snapshot ja mittarit
- Vaihe 2: laajennettu seulonta, PDF-jäsentäminen ja raportointi automatisoituna

Aja `make setup` (tai toteuta vastaavat komennot käsin) ennen ensimmäistä muutosta ja varmista pre-commit-koukut komennolla `pre-commit run --all-files`.
