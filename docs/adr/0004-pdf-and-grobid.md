# ADR 0004, PDF-haku ja GROBID-jäsentäminen

Päiväys, 2025-10-01
Status, Hyväksytty

Konteksti
- Vaiheen 4 DoD vaatii täyden tekstin latauksen ja kevyen jäsentämisen jatkoprosesseja varten
- ArXiv ja PubMed Central tarjoavat suorat PDF-linkit, mutta DOI-pohjaiset artikkelit edellyttävät OA-välittäjää
- GROBID on de facto työkalu TEI-XML rakenteen tuottamiseen ja siitä johdettuun tiivistelmään

Päätös
- Toteutetaan `la pdf`, joka yrittää latauksen järjestyksessä arXiv → PMC → Unpaywall; jokainen yritys kirjataan audit CSV:hen
- Säilytetään kaikki ladatut PDF:t hakemistossa `data/pdfs/` ja talletetaan lisenssi sekä tiedoston polku Parquet-sarakkeisiin (`pdf_path`, `pdf_license`, `has_fulltext`)
- Otetaan käyttöön GROBID Docker-palvelu (paikallisesti portissa 8070) ja `la parse`, joka kirjoittaa TEI-XML:n sekä tiivistetyn Markdown-tekstin hakemistoon `data/parsed/<id>/`
- Jäsentämisen virheet kerätään `data/logs/parse_errors.csv` tiedostoon ja CLI sallii `--sample` rajauksen nopeisiin kokeiluihin

Seuraukset
- Nightly-ajo voi tarkistaa PDF-kattavuuden (`has_fulltext`) ja onnistuneiden parsejen osuuden (`parsed_ok`) ilman käsityötä
- Audit-loki mahdollistaa latausvirheiden ja OA-lisenssien läpinäkyvän raportoinnin; 429/503 uudelleenyritykset ovat konfiguroitavissa
- TEI-säilö pitää lähdemetadatan ja viitelistat myöhempää RAG- tai raportointikäyttöä varten; Markdown helpottaa nopeaa jatkokäsittelyä

Linkit
- src/la_pkg/pdf/download.py
- src/la_pkg/parse/run.py
- src/la_pkg/cli.py
- .github/workflows/e2e-nightly.yml
