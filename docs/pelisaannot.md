# Yhteiset pelisäännöt

## Periaatteet
- Hyödynnä repositoryn juuren `docs`-kansiota yhtenäisenä totuuden lähteenä
- Tee päätöksistä ADR-merkintä; lisää uudet tiedostot `docs/adr`-kansioon
- Pidä näkyvyys ajan tasalla: projektitaulu (Backlog, Doing, Review, Done) ja kalenterimuistio perjantain porttikokoukseen (30 min)

## Työtapa
- Trunk-based-virta: työ aloitetaan feature-haaralla ja tuodaan PR:n kautta `main`iin
- PR vaatii tarkastuksen ja vihreän CI-ajon; squash-merge on oletus
- Aja `make lint` ja `make test` tai vastaavat komennot ennen PR:ää

## Artefaktit ja ympäristö
- Kehitys tapahtuu `.venv`-ympäristössä, paketit kirjataan `requirements.txt`iin
- Keskeneräiset datat, cachet ja muut välitulokset pidetään versionhallinnan ulkopuolella (käytä `data/`, `cache/`, `output/`-hakemistoja)
- Konfiguraatiot dokumentoidaan `config.example.toml`- ja `.env.example`-tiedostoissa

## Laatuportit ja mittarit
- Seuraa mittarikynnysarvoja (`docs/mittarit.md`): deduplikointi < 5 %, PDF-parsinnan onnistuminen ≥ 80 %, seulonnan recall ≥ 0.9
- CI saa epäonnistua vain, jos mittarikynnys rikkoutuu – tällöin aloita juurisyyn analyysi porttikokouksessa

## Commit-viestit
- Käytä Conventional Commits -muotoa (`<type>(<scope>): <kuvaus>`)
- Yleisimmät tyypit: `feat`, `fix`, `chore`, `docs`, `refactor`, `test`
- Esimerkki: `feat(cli): lisää la hello`

## Dokumentaatio
- Päivitä `README.md` lyhyeksi yhteenvedoksi ja linkitä tarkemmat dokumentit
- Arkkitehtuurin päälinjat ovat `docs/ARCHITECTURE.md`-tiedostossa (yksi sivu)
- Päivitä ADR:t muutoksista; status "Hyväksytty" kun päätös on voimassa

## Definition of Ready
- Tavoite on kuvattu selkeästi ja rajattu yhteen lopputulokseen
- Testitapaus tai hyväksymiskriteeri on tunnistettu
- Tarvittavat data- ja integraatiolähteet ovat saatavilla tai mockattu

## Definition of Done
- Automaattiset testit ja lintterit ovat vihreitä
- Mittarit tulostuvat ja täyttävät kynnysarvonsa
- Dokumentaatio (README, ADR, pelisäännöt) on päivitetty tarvittaessa
- CI on vihreä ja muutokselle on hyväksytty PR
