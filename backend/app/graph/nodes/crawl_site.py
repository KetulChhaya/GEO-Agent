"""crawl_site node (Phase 1, task 1.1).

Given a company URL in the audit state, discover and fetch up to MAX_PAGES pages
from that site, strip boilerplate, and return the extracted main content plus a
detected brand name.

Design notes (this repo is read by interviewers, so keep it boring and readable):

* This is a LangGraph node. A node is just an async function that takes the state
  and returns a dict of the fields to update -- here `{"pages": ..., "brand_name":
  ..., "errors": ...}`. The graph itself is assembled later (task 1.4); this file
  only defines the node so it can be unit-exercised in isolation.
* Recoverable problems (a page 404s, robots.txt is missing, a sitemap is malformed)
  never raise -- they append a human-readable line to `errors` and carry on. A
  single bad page must not kill the whole audit.
* No LLM calls happen here: crawling is a pre-LLM step, so there is no ProviderClient
  and nothing to bill against the token budget.
"""

import asyncio
import xml.etree.ElementTree as ET
from typing import Any
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import httpx
from bs4 import BeautifulSoup

from app.config import get_settings
from app.graph.state import AuditState, PageMeta

# A descriptive User-Agent is polite and makes our crawler identifiable in logs.
USER_AGENT = "CiteSightBot/0.1 (+https://github.com/KetulChhaya; AI visibility audit)"

# Tags whose text is navigation/chrome, not the page's actual content.
_BOILERPLATE_TAGS = ["script", "style", "nav", "footer", "header", "aside", "noscript"]

# Bound how many pages we fetch at once so we do not hammer the target site.
_FETCH_CONCURRENCY = 5

# How deep the fallback link crawl follows same-domain links from the homepage.
_LINK_CRAWL_DEPTH = 2

# A page needs at least this many words to count as "usable" content. The graph's
# conditional edge (task 1.2) fails the run when fewer than 3 usable pages exist.
MIN_USABLE_WORDS = 50
MIN_USABLE_PAGES = 3


def _origin(url: str) -> str:
    """Scheme + host, e.g. https://example.com -- the base for robots/sitemap."""
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}"


def _same_domain(url: str, base_url: str) -> bool:
    return urlparse(url).netloc == urlparse(base_url).netloc


async def _load_robots(
    client: httpx.AsyncClient, origin: str, errors: list[str]
) -> RobotFileParser:
    """Fetch and parse robots.txt.

    A fresh RobotFileParser (``last_checked == 0``) allows everything, so on a
    missing/unreadable robots.txt we just return it untouched -- no restrictions.
    On success we parse the rules AND call ``modified()``: can_fetch() ignores
    parsed rules until last_checked is set, so this step is what makes disallow
    directives actually take effect.
    """
    parser = RobotFileParser()
    robots_url = urljoin(origin + "/", "robots.txt")
    try:
        resp = await client.get(robots_url)
        if resp.status_code == 200:
            parser.parse(resp.text.splitlines())
            parser.modified()
    except httpx.HTTPError as exc:
        errors.append(f"robots.txt fetch failed ({robots_url}): {exc}; assuming allow-all")
    return parser


def _sitemap_locs(xml_text: str) -> tuple[list[str], list[str]]:
    """Parse a sitemap or sitemap-index. Returns (page_urls, nested_sitemap_urls).

    A <urlset> lists page <loc>s; a <sitemapindex> lists child sitemap <loc>s. We
    ignore the namespace prefix by matching on the local tag name."""
    page_urls: list[str] = []
    nested: list[str] = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return page_urls, nested

    root_tag = root.tag.rsplit("}", 1)[-1]  # strip "{namespace}" prefix
    target = nested if root_tag == "sitemapindex" else page_urls
    for loc in root.iter():
        if loc.tag.rsplit("}", 1)[-1] == "loc" and loc.text:
            target.append(loc.text.strip())
    return page_urls, nested


async def _urls_from_sitemaps(
    client: httpx.AsyncClient, sitemap_urls: list[str], base_url: str
) -> list[str]:
    """Collect same-domain page URLs from the given sitemaps, following one level
    of sitemap-index nesting."""
    found: list[str] = []
    for sitemap_url in sitemap_urls:
        try:
            resp = await client.get(sitemap_url)
        except httpx.HTTPError:
            continue
        if resp.status_code != 200:
            continue
        pages, nested = _sitemap_locs(resp.text)
        found.extend(pages)
        for child in nested:  # one level deep only, to bound work
            try:
                child_resp = await client.get(child)
            except httpx.HTTPError:
                continue
            if child_resp.status_code == 200:
                child_pages, _ = _sitemap_locs(child_resp.text)
                found.extend(child_pages)
    return [u for u in found if _same_domain(u, base_url)]


def _links_from_html(html: str, page_url: str, base_url: str) -> list[str]:
    """Same-domain hrefs on a page, resolved to absolute URLs and de-fragmented."""
    soup = BeautifulSoup(html, "html.parser")
    links: list[str] = []
    for a in soup.find_all("a", href=True):
        absolute = urljoin(page_url, str(a["href"]))
        absolute = absolute.split("#", 1)[0]  # drop in-page anchors
        if absolute and _same_domain(absolute, base_url):
            links.append(absolute)
    return links


async def _link_crawl(
    client: httpx.AsyncClient, base_url: str, robots: RobotFileParser, max_pages: int
) -> list[str]:
    """Fallback discovery: breadth-first walk of same-domain links from the
    homepage, up to _LINK_CRAWL_DEPTH hops, stopping at max_pages."""
    discovered: list[str] = []
    seen: set[str] = set()
    frontier = [base_url]
    for _ in range(_LINK_CRAWL_DEPTH + 1):
        next_frontier: list[str] = []
        for url in frontier:
            if url in seen or len(discovered) >= max_pages:
                continue
            seen.add(url)
            if not robots.can_fetch(USER_AGENT, url):
                continue
            discovered.append(url)
            try:
                resp = await client.get(url)
            except httpx.HTTPError:
                continue
            if resp.status_code != 200 or "html" not in resp.headers.get("content-type", ""):
                continue
            for link in _links_from_html(resp.text, url, base_url):
                if link not in seen:
                    next_frontier.append(link)
        frontier = next_frontier
        if not frontier or len(discovered) >= max_pages:
            break
    return discovered


async def _discover_urls(
    client: httpx.AsyncClient, base_url: str, robots: RobotFileParser, max_pages: int
) -> list[str]:
    """Find candidate page URLs: sitemap first, link-crawl fallback. Result is
    robots-allowed, de-duplicated (order preserved), and capped at max_pages."""
    origin = _origin(base_url)
    sitemap_urls = list(robots.site_maps() or [])
    if not sitemap_urls:
        sitemap_urls = [urljoin(origin + "/", "sitemap.xml")]

    candidates = await _urls_from_sitemaps(client, sitemap_urls, base_url)

    # Fall back to a link crawl when the sitemap gave us too little to work with.
    if len(candidates) < 3:
        candidates = await _link_crawl(client, base_url, robots, max_pages)

    # Always ensure the homepage itself is in the set.
    if base_url not in candidates:
        candidates.insert(0, base_url)

    allowed = [u for u in candidates if robots.can_fetch(USER_AGENT, u)]

    deduped: list[str] = []
    seen: set[str] = set()
    for url in allowed:
        if url not in seen:
            seen.add(url)
            deduped.append(url)
    return deduped[:max_pages]


def _extract_main(html: str) -> tuple[str | None, str]:
    """Return (title, main_text) with boilerplate tags removed and whitespace
    collapsed."""
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(strip=True) if soup.title else None
    for tag in soup(_BOILERPLATE_TAGS):
        tag.decompose()
    # Pull text from <body> only so <head> metadata (title, meta tags) does not
    # leak into the page's main content.
    body = soup.body or soup
    text = body.get_text(separator=" ", strip=True)
    text = " ".join(text.split())  # collapse runs of whitespace/newlines
    return title, text


def _extract_brand(html: str) -> str | None:
    """Best-effort brand name: og:site_name, then og:title, then <title>."""
    soup = BeautifulSoup(html, "html.parser")
    for prop in ("og:site_name", "og:title"):
        tag = soup.find("meta", attrs={"property": prop})
        if tag and tag.get("content"):
            return str(tag["content"]).strip()
    if soup.title and soup.title.string:
        return soup.title.string.strip()
    return None


async def _fetch_page(
    client: httpx.AsyncClient, url: str, sem: asyncio.Semaphore, errors: list[str]
) -> tuple[str, str] | None:
    """Fetch one page. Returns (url, html) or None (with a note in errors) if it
    is not a 2xx HTML response."""
    async with sem:
        try:
            resp = await client.get(url)
        except httpx.HTTPError as exc:
            errors.append(f"fetch failed ({url}): {exc}")
            return None
    if not (200 <= resp.status_code < 300):
        errors.append(f"skip {url}: HTTP {resp.status_code}")
        return None
    if "html" not in resp.headers.get("content-type", ""):
        errors.append(f"skip {url}: non-HTML content-type")
        return None
    return url, resp.text


async def crawl_site(state: AuditState) -> dict[str, Any]:
    """Crawl the site at `state.url`, returning extracted pages and brand name.

    Returns a dict of state updates for LangGraph to merge: `pages`, `brand_name`,
    and the accumulated `errors`.
    """
    settings = get_settings()
    max_pages = settings.max_pages
    errors: list[str] = list(state.errors)  # copy so we append, not mutate input
    base_url = state.url
    origin = _origin(base_url)

    async with httpx.AsyncClient(
        headers={"User-Agent": USER_AGENT},
        follow_redirects=True,
        timeout=15.0,
    ) as client:
        robots = await _load_robots(client, origin, errors)
        urls = await _discover_urls(client, base_url, robots, max_pages)

        sem = asyncio.Semaphore(_FETCH_CONCURRENCY)
        results = await asyncio.gather(
            *(_fetch_page(client, url, sem, errors) for url in urls)
        )

    pages: list[PageMeta] = []
    brand_name = state.brand_name
    for fetched in results:
        if fetched is None:
            continue
        url, html = fetched
        title, content = _extract_main(html)
        if not content:  # nothing usable (e.g. a JS-only shell) -- skip
            continue
        pages.append(
            PageMeta(url=url, title=title, content=content, word_count=len(content.split()))
        )
        # Derive the brand from the homepage specifically, when we reach it.
        if brand_name is None and url in (base_url, origin, origin + "/"):
            brand_name = _extract_brand(html)

    # If we never matched the homepage exactly but have pages, fall back to the
    # first page's brand so the field is populated when content exists.
    if brand_name is None and pages:
        brand_name = pages[0].title

    # Task 1.2 gate: a run needs >= MIN_USABLE_PAGES pages with real content. A
    # JS-only site (empty shells, no extractable text) trips this and the graph's
    # conditional edge routes straight to END. We set the status here because the
    # routing function cannot mutate state -- it only reads this decision.
    updates: dict[str, Any] = {"pages": pages, "brand_name": brand_name, "errors": errors}
    usable = sum(1 for p in pages if p.word_count >= MIN_USABLE_WORDS)
    if usable < MIN_USABLE_PAGES:
        errors.append(
            f"insufficient content: found {usable} usable page(s) "
            f"(>= {MIN_USABLE_WORDS} words), need {MIN_USABLE_PAGES}"
        )
        updates["status"] = "failed_insufficient_content"
    return updates
