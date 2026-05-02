#!/usr/bin/env python3
"""Quick BibTeX reference verifier.

Checks references.bib against free metadata APIs:
  - arXiv for arXiv IDs
  - Crossref for exact DOI metadata
  - Semantic Scholar for fast CS/AI title fallback
  - OpenAlex/arXiv title fallback
  - Crossref title search in --deep mode

The script is intentionally dependency-free so it can run in a clean Python
environment. It reports likely mismatches; it does not edit the .bib file.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import unicodedata
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any
from concurrent.futures import ThreadPoolExecutor, as_completed


USER_AGENT = "reference-verifier/0.1 (mailto:example@example.com)"
SEMANTIC_SCHOLAR_API_KEY = os.environ.get("SEMANTIC_SCHOLAR_API_KEY") or os.environ.get("S2_API_KEY")
ARXIV_RE = re.compile(r"arxiv[:/\s]*([0-9]{4}\.[0-9]{4,5})(?:v[0-9]+)?", re.I)
DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.I)


@dataclass
class Entry:
    key: str
    kind: str
    fields: dict[str, str]


@dataclass
class Candidate:
    source: str
    title: str = ""
    authors: list[str] | None = None
    year: str = ""
    venue: str = ""
    doi: str = ""
    url: str = ""


def strip_latex(value: str) -> str:
    value = re.sub(r"\\url\{([^{}]+)\}", r"\1", value)
    value = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?(?:\{([^{}]*)\})?", r"\1", value)
    value = value.replace("{", "").replace("}", "")
    value = value.replace("\\&", "&").replace("\\_", "_")
    value = value.replace("\\`", "").replace("\\'", "").replace('\\"', "")
    value = value.replace("\\ss", "ss")
    return re.sub(r"\s+", " ", value).strip()


def norm_text(value: str) -> str:
    value = strip_latex(value).lower()
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    value = value.replace("behaviours", "behaviors")
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def title_score(left: str, right: str) -> float:
    return SequenceMatcher(None, norm_text(left), norm_text(right)).ratio()


def parse_bib(path: Path) -> list[Entry]:
    text = path.read_text(encoding="utf-8")
    entries: list[Entry] = []
    i = 0
    while True:
        at = text.find("@", i)
        if at < 0:
            break
        m = re.match(r"@([A-Za-z]+)\s*\{\s*([^,]+),", text[at:])
        if not m:
            i = at + 1
            continue
        kind, key = m.group(1), m.group(2).strip()
        body_start = at + m.end()
        depth = 1
        j = body_start
        while j < len(text) and depth:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
            j += 1
        body = text[body_start : j - 1]
        entries.append(Entry(key=key, kind=kind.lower(), fields=parse_fields(body)))
        i = j
    return entries


def parse_fields(body: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    i = 0
    while i < len(body):
        m = re.search(r"([A-Za-z][A-Za-z0-9_-]*)\s*=", body[i:])
        if not m:
            break
        name = m.group(1).lower()
        pos = i + m.end()
        while pos < len(body) and body[pos].isspace():
            pos += 1
        if pos >= len(body):
            break
        if body[pos] == "{":
            depth = 1
            pos += 1
            start = pos
            while pos < len(body) and depth:
                if body[pos] == "{":
                    depth += 1
                elif body[pos] == "}":
                    depth -= 1
                pos += 1
            value = body[start : pos - 1]
        elif body[pos] == '"':
            pos += 1
            start = pos
            while pos < len(body) and body[pos] != '"':
                pos += 1
            value = body[start:pos]
            pos += 1
        else:
            start = pos
            while pos < len(body) and body[pos] not in ",\n":
                pos += 1
            value = body[start:pos]
        fields[name] = re.sub(r"\s+", " ", value).strip()
        i = pos + 1
    return fields


def http_json(url: str, headers: dict[str, str] | None = None) -> dict[str, Any] | None:
    request_headers = {"User-Agent": USER_AGENT}
    if headers:
        request_headers.update(headers)
    req = urllib.request.Request(url, headers=request_headers)
    try:
        with urllib.request.urlopen(req, timeout=5) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception:
        return None


def http_xml(url: str) -> ET.Element | None:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=5) as response:
            return ET.fromstring(response.read())
    except Exception:
        return None


def first_year(value: str) -> str:
    m = re.search(r"\b(19|20)\d{2}\b", value or "")
    return m.group(0) if m else ""


def author_last(name: str) -> str:
    name = strip_latex(name)
    if "," in name:
        name = name.split(",", 1)[0]
    else:
        name = name.split()[-1] if name.split() else ""
    return norm_text(name)


def bib_authors(entry: Entry) -> list[str]:
    author = entry.fields.get("author", "")
    return [a.strip() for a in re.split(r"\s+and\s+", author) if a.strip()]


def entry_text(entry: Entry) -> str:
    return " ".join(entry.fields.values())


def arxiv_id(entry: Entry) -> str:
    m = ARXIV_RE.search(entry_text(entry))
    return m.group(1) if m else ""


def doi(entry: Entry) -> str:
    if "doi" in entry.fields:
        return strip_latex(entry.fields["doi"])
    m = DOI_RE.search(entry_text(entry))
    return m.group(0) if m else ""


def crossref_by_doi(identifier: str) -> Candidate | None:
    url = "https://api.crossref.org/works/" + urllib.parse.quote(identifier)
    data = http_json(url)
    msg = (data or {}).get("message", {})
    if not msg:
        return None
    return crossref_candidate(msg)


def crossref_by_title(title: str) -> list[Candidate]:
    q = urllib.parse.urlencode({"query.bibliographic": title, "rows": "3"})
    data = http_json("https://api.crossref.org/works?" + q)
    items = ((data or {}).get("message") or {}).get("items") or []
    return [crossref_candidate(item) for item in items]


def crossref_candidate(item: dict[str, Any]) -> Candidate:
    authors = [
        " ".join(filter(None, [a.get("given", ""), a.get("family", "")]))
        for a in item.get("author", [])
    ]
    year_parts = item.get("published-print") or item.get("published-online") or item.get("created") or {}
    year = str((year_parts.get("date-parts") or [[""]])[0][0])
    return Candidate(
        source="crossref",
        title=(item.get("title") or [""])[0],
        authors=authors,
        year=year,
        venue=(item.get("container-title") or [""])[0],
        doi=item.get("DOI", ""),
        url=item.get("URL", ""),
    )


def semantic_by_id(prefix: str, identifier: str) -> Candidate | None:
    fields = "title,authors,year,venue,externalIds,url"
    url = f"https://api.semanticscholar.org/graph/v1/paper/{prefix}:{urllib.parse.quote(identifier)}?fields={fields}"
    data = http_json(url, semantic_headers())
    return semantic_candidate(data) if data and data.get("title") else None


def semantic_by_title(title: str) -> list[Candidate]:
    q = urllib.parse.urlencode({"query": title, "limit": "3", "fields": "title,authors,year,venue,externalIds,url"})
    data = http_json("https://api.semanticscholar.org/graph/v1/paper/search?" + q, semantic_headers())
    return [semantic_candidate(item) for item in (data or {}).get("data", [])]


def semantic_headers() -> dict[str, str]:
    return {"x-api-key": SEMANTIC_SCHOLAR_API_KEY} if SEMANTIC_SCHOLAR_API_KEY else {}


def semantic_candidate(item: dict[str, Any]) -> Candidate:
    ext = item.get("externalIds") or {}
    return Candidate(
        source="semantic_scholar",
        title=item.get("title", ""),
        authors=[a.get("name", "") for a in item.get("authors", [])],
        year=str(item.get("year") or ""),
        venue=item.get("venue") or "",
        doi=ext.get("DOI", ""),
        url=item.get("url") or "",
    )


def openalex_by_title(title: str) -> list[Candidate]:
    q = urllib.parse.urlencode({"search": title, "per-page": "3"})
    data = http_json("https://api.openalex.org/works?" + q)
    return [openalex_candidate(item) for item in (data or {}).get("results", [])]


def openalex_candidate(item: dict[str, Any]) -> Candidate:
    authors = [
        (((a.get("author") or {}).get("display_name")) or "")
        for a in item.get("authorships", [])
    ]
    venue = ((item.get("primary_location") or {}).get("source") or {}).get("display_name") or ""
    return Candidate(
        source="openalex",
        title=item.get("title", ""),
        authors=authors,
        year=str(item.get("publication_year") or ""),
        venue=venue,
        doi=(item.get("doi") or "").replace("https://doi.org/", ""),
        url=item.get("id") or "",
    )


def arxiv_by_id(identifier: str) -> Candidate | None:
    root = http_xml("https://export.arxiv.org/api/query?id_list=" + urllib.parse.quote(identifier))
    return arxiv_first_candidate(root)


def arxiv_by_title(title: str) -> Candidate | None:
    root = http_xml("https://export.arxiv.org/api/query?max_results=3&search_query=ti:" + urllib.parse.quote(f'"{title}"'))
    return arxiv_first_candidate(root, wanted_title=title)


def arxiv_first_candidate(root: ET.Element | None, wanted_title: str = "") -> Candidate | None:
    if root is None:
        return None
    ns = {"a": "http://www.w3.org/2005/Atom"}
    best: Candidate | None = None
    best_score = 0.0
    for e in root.findall("a:entry", ns):
        title = re.sub(r"\s+", " ", (e.findtext("a:title", "", ns) or "")).strip()
        authors = [a.findtext("a:name", "", ns) or "" for a in e.findall("a:author", ns)]
        published = e.findtext("a:published", "", ns) or ""
        cand = Candidate(
            source="arxiv",
            title=title,
            authors=authors,
            year=first_year(published),
            venue="arXiv",
            url=e.findtext("a:id", "", ns) or "",
        )
        score = title_score(wanted_title, title) if wanted_title else 1.0
        if score > best_score:
            best, best_score = cand, score
    return best


def official_web_reference(entry: Entry) -> bool:
    text = entry_text(entry).lower()
    return entry.kind == "misc" and any(host in text for host in [
        "qwenlm.github.io",
        "mistral.ai",
        "huggingface.co",
    ])


def find_candidates(entry: Entry, polite_sleep: float, deep: bool) -> list[Candidate]:
    title = strip_latex(entry.fields.get("title", ""))
    candidates: list[Candidate] = []
    if official_web_reference(entry):
        return [Candidate(source="manual_web", title=title)]
    identifier = arxiv_id(entry)
    identifier_doi = doi(entry)
    if identifier:
        for cand in [arxiv_by_id(identifier), semantic_by_id("ARXIV", identifier)]:
            if cand:
                candidates.append(cand)
        time.sleep(polite_sleep)
        return candidates
    if identifier_doi:
        for cand in [crossref_by_doi(identifier_doi), semantic_by_id("DOI", identifier_doi)]:
            if cand:
                candidates.append(cand)
        time.sleep(polite_sleep)
        return candidates
    arxiv_cand = arxiv_by_title(title)
    if arxiv_cand:
        candidates.append(arxiv_cand)
    candidates.extend(semantic_by_title(title))
    candidates.extend(openalex_by_title(title))
    if deep:
        candidates.extend(crossref_by_title(title))
    time.sleep(polite_sleep)
    return candidates


def best_candidate(entry: Entry, candidates: list[Candidate]) -> tuple[Candidate | None, float]:
    title = strip_latex(entry.fields.get("title", ""))
    best = None
    score = 0.0
    for cand in candidates:
        s = title_score(title, cand.title)
        if s > score:
            best, score = cand, s
    return best, score


def compare(entry: Entry, cand: Candidate | None, score: float) -> list[str]:
    if cand is None:
        if official_web_reference(entry):
            return ["manual-web: official web/model-card reference; no scholarly API match expected"]
        return ["not-found: no API candidate above threshold"]
    issues: list[str] = []
    title = strip_latex(entry.fields.get("title", ""))
    if score < 0.92:
        issues.append(f"title? bib={title!r} api={cand.title!r} score={score:.2f}")
    bib_year = strip_latex(entry.fields.get("year", ""))
    bib_uses_arxiv = "arxiv" in norm_text(entry.fields.get("journal", "") + " " + entry.fields.get("booktitle", ""))
    if bib_year and cand.year and bib_year != cand.year and (cand.source != "arxiv" or bib_uses_arxiv):
        issues.append(f"year? bib={bib_year} api={cand.year}")
    b_authors = bib_authors(entry)
    if b_authors and cand.authors:
        bib_first = author_last(b_authors[0])
        api_first = author_last(cand.authors[0])
        corporate = bib_first in {"qwen team", "qwen", "gemma team", "gemma", "deepseek ai", "meta ai", "mistral ai"}
        corporate = corporate or api_first in {"qwen team", "qwen", "gemma team", "gemma", "deepseek ai"}
        if bib_first != api_first and not corporate:
            issues.append(f"first-author? bib={strip_latex(b_authors[0])!r} api={cand.authors[0]!r}")
    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bib", nargs="?", default="references.bib")
    parser.add_argument("--strict", action="store_true", help="exit non-zero on warnings")
    parser.add_argument("--deep", action="store_true", help="also run slower Crossref/OpenAlex/arXiv title fallbacks")
    parser.add_argument("--sleep", type=float, default=0.0, help="delay after each entry lookup")
    parser.add_argument("--workers", type=int, default=6, help="parallel API lookup workers")
    parser.add_argument("--limit", type=int, default=0, help="check only first N entries")
    args = parser.parse_args()

    entries = parse_bib(Path(args.bib))
    if args.limit:
        entries = entries[: args.limit]

    def check_one(index_entry: tuple[int, Entry]) -> tuple[int, Entry, Candidate | None, float, list[str]]:
        index, entry = index_entry
        candidates = find_candidates(entry, args.sleep, args.deep)
        cand, score = best_candidate(entry, candidates)
        return index, entry, cand, score, compare(entry, cand, score)

    results: list[tuple[int, Entry, Candidate | None, float, list[str]] | None] = [None] * len(entries)
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(check_one, item) for item in enumerate(entries)]
        for future in as_completed(futures):
            index, entry, cand, score, issues = future.result()
            results[index] = (index, entry, cand, score, issues)

    warnings = 0
    print(f"Checking {len(entries)} references from {args.bib}\n", flush=True)
    for result in results:
        assert result is not None
        _, entry, cand, _score, issues = result
        status = "OK" if not issues else "WARN"
        if issues:
            warnings += 1
        matched = f"{cand.source}:{cand.title[:70]}" if cand else "no-match"
        print(f"[{status}] {entry.key} -> {matched}", flush=True)
        for issue in issues:
            print(f"  - {issue}", flush=True)
    print(f"\nDone: {len(entries) - warnings} OK, {warnings} warnings", flush=True)
    return 1 if args.strict and warnings else 0


if __name__ == "__main__":
    sys.exit(main())
