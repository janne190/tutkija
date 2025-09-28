"""Utilities for extracting metadata from TEI XML."""

from __future__ import annotations

from defusedxml import ElementTree as ET

NS = {"tei": "http://www.tei-c.org/ns/1.0"}


def _text_or_empty(elements: list[ET.Element]) -> str:
    if not elements:
        return ""
    parts: list[str] = []
    for element in elements:
        parts.extend(list(element.itertext()))
    return " ".join(part.strip() for part in parts if part.strip()).strip()


def tei_to_title_abstract_refs(tei_xml: str) -> dict[str, object]:
    """Extract lightweight fields from a TEI XML string."""

    root = ET.fromstring(tei_xml.encode("utf-8"))
    titles = root.findall(".//tei:titleStmt/tei:title", NS)
    title = _text_or_empty(titles)
    abstracts = root.findall(".//tei:profileDesc/tei:abstract", NS)
    abstract = _text_or_empty(abstracts)

    refs: list[str] = []
    for bibl in root.findall(".//tei:listBibl/tei:biblStruct", NS):
        snippet = "".join(text.strip() for text in bibl.itertext()).strip()
        if snippet:
            refs.append(snippet[:200])

    return {"title": title, "abstract": abstract, "n_refs": len(refs)}
