# ADR 0003, Screening-malli ja fallback-strategiat

Päiväys, 2025-09-25
Status, Hyväksytty

Konteksti
- Vaiheen 3 DoD edellyttää, että `la screen` tuottaa automaattiset priorisoinnit ilman pakollista kultaista opetusdataa
- Pipeline yhdistää sääntöpohjaisen esiseulonnan ja mallipohjaisen pisteytyksen; tulokset pitää kirjata PRISMA-tyylisiin lokitietoihin

Päätös
- Oletusmoottori on scikit-learn: TF-IDF (1-2 gram) + logistinen regressio, satunnaissiemen deterministisessä tilassa
- Jos kultaa on saatavilla, valitaan todennäköisyyskynnys iteratiivisesti siten, että saavutetaan vähintään pyydetty recall
- Jos kultaa ei ole mutta siemenartikkeleita annetaan, lasketaan TF-IDF + kosinietäisyys ja kasvatetaan todennäköisyyksiä `0.5 + 0.5 * max_similarity`
- Jos kumpaakaan ei ole, annetaan tasaprobabiliteetti 0.5, jotta kaikki kirjaukset etenevät manuaaliseen läpikäyntiin
- Kaikki syyt tallennetaan listana hyväksytyillä avaimilla (`language filter`, `year filter`, `type filter`, `manual check`) ja sisällytettyjen rivien syyt tyhjennetään ennen kirjoitusta
- `screen_log.csv` sisältää aina PRISMA-luvut, käytetyn moottorin, siemenmäärän, satunnaissiementä vastaavan `random_state`:n, kynnyksen, fallback-strategian ja tuotetun tiedoston polun
- ASReview on valinnainen: jos kirjastoa ei ole asennettu, CLI ohjeistaa asentamaan `tutkija[asreview]` tai ajamaan scikit-versiolla

Seuraukset
- Ilman kultaa pipeline pysyy deterministisenä ja avoimena tarkistettaville; siemenet tarjoavat kevyen tavan priorisoida aiheeseen liittyviä julkaisuja
- Lokit mahdollistavat auditoinnin (≥70 % käsittelystä, täydelliset kentät) ja tukevat yöajon sekä PowerShell-auditin tarkastuksia
- Yhteiset syyavaimet ja tyhjät listat sisällytetyille riville helpottavat käyttöliittymien rakentamista ja raportointia

Linkit
- src/la_pkg/screening/model.py
- src/la_pkg/cli.py
- scripts/audit.ps1
- .github/workflows/e2e-nightly.yml
