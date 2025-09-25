# ADR 0001, Työkalupino

Päiväys, 2025-09-25
Status, Hyväksytty

Konteksti
- Tarvitsemme yhteisen pinon paikalliseen kehitykseen ja CI-putkeen

Päätös
- Python 3.11, hallinta `uv`:lla tai Poetrylla tilanteen mukaan
- Lint ja tyyppitarkistus: Ruff + MyPy
- Testaus: Pytest
- Paikalliset tarkistukset: pre-commit (Ruff, mypy, perushooks)
- CI: GitHub Actions (lint, testit, savutesti)

Seuraukset
- Yhdenmukainen kehitysympäristö ja nopea palaute virheistä
- Työkalupino ohjaa commit- ja PR-käytänteet automaattisiksi

Linkit
- docs/pelisaannot.md
- .github/workflows/
