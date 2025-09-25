# Tutkija-arkkitehtuuri

## Konteksti
- Tutkija automatisoi kirjallisuuskatsauksen vaiheet: haku, seulonta, lukeminen ja raportointi
- Projektia ajetaan paikallisesti tai CI-putkessa; ulkoiset integraatiot rajautuvat hakurajapintoihin ja tiedontallennukseen

## Pääkomponentit
- **Orkestrointi**: moniagenttinen työnkulku, joka koordinoi vaiheita ja vastaa ajastuslogiikasta
- **Haku & metatieto**: rajapinnat (esim. OpenAlex) ja tulosten normalisointi yhteiseen skeemaan
- **Seulonta**: sääntö- ja mallipohjainen tarkistus, joka karsii pois epärelevantit julkaisut
- **PDF-jäsentäminen**: lataus, tekstin purku ja segmentointi jatkokäsittelyä varten
- **RAG & tositteet**: vektori-indeksointi ja vastausten perusteleminen lähdeviitteillä
- **Kirjoitus & raportointi**: koostaa löydökset yhdeksi raportiksi ja tuottaa PRISMA-kaavion
- **Lokitus & mittarit**: keskeiset mittarit raportoidaan `docs/mittarit.md` -kynnysarvoja vasten

## Tietovirta
1. Aihe ja rajaukset syötetään CLI:n tai konfiguraation kautta
2. Haku kerää perus- ja täydentävän metadatan, jonka jälkeen tulokset deduplikoidaan
3. Seulonta ja PDF-jäsentäminen jalostavat aineiston jatkoanalyyseihin
4. RAG tuottaa vastaukset ja viitteet, jotka kirjataan raporttiin
5. Mittarit ja lokit päivitetään ja julkaistaan CI-raporteissa

## Laatu ja operointi
- Jokainen vaihe tarjoaa mittarit (recall, onnistumisprosentit, latenssi)
- CI pipeline ajaa `ruff`, `mypy`, `pytest` ja savutestin (`la hello`)
- Porttikokous arvioi mittarien trendiä ja päättää jatkotoimenpiteistä

## Jatkokehitys
- Vaihe 1 keskittyy hakumoduulin tuotantokelpoistamiseen ja golden snapshot -testiin
- Tulevissa vaiheissa laajennetaan datalähteitä ja automatisoidaan julkaisuprosessi
