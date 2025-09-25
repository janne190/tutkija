# ADR 0002, Versionointi ja julkaisu

Päiväys, 2025-09-25
Status, Hyväksytty

Konteksti
- Tarvitsemme johdonmukaisen versionointimallin ja julkaisutavan, jotta artefaktit voidaan toistaa

Päätös
- Noudatetaan SemVer-versiointia (`MAJOR.MINOR.PATCH`)
- Jokainen tuotantojulkaisu syntyy Git-tagista (`vMAJOR.MINOR.PATCH`)
- CI:n release-workflow rakentaa paketin ja liittää mittarit julkaisutietoihin
- Paketointi ja jakelu tehdään `uv`/`pip`-yhteensopivana lähiaikataulussa

Seuraukset
- Julkaisujen audit trail on selkeä ja palautettavissa tagien perusteella
- Patch-julkaisut priorisoivat bugikorjaukset, minor-julkaisut uusia ominaisuuksia; major vaatii porttikokouksen päätöksen

Linkit
- docs/pelisaannot.md
- README.md
