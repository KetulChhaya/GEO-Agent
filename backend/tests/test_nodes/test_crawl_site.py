"""crawl_site tests (task 1.6) -- no network.

We monkeypatch `crawl_site._make_client` (the one place the HTTP client is built)
to return an httpx.AsyncClient wired to an httpx.MockTransport. The transport's
handler answers each request from an in-memory fixture site, so the crawler runs
its full robots/sitemap/link-crawl logic without touching a real network.
"""

import os

# crawl_site() reads settings.max_pages at runtime; give a value so get_settings()
# validates. No DB connection is made here -- crawling is DB-free.
os.environ.setdefault(
    "DATABASE_URL", "postgresql+asyncpg://citesight:citesight@localhost:5433/citesight"
)

import httpx  # noqa: E402
import pytest  # noqa: E402

import app.graph.nodes.crawl_site as crawl_mod  # noqa: E402
from app.graph.nodes.crawl_site import crawl_site  # noqa: E402
from app.graph.state import AuditState  # noqa: E402

BASE = "http://acme.test/"

# Each page has unique content (> 50 words to count as "usable" per MIN_USABLE_WORDS).

HOME_CONTENT = (
    "Acme builds industrial widgets for factories across the world and has shipped "
    "durable equipment since nineteen ninety. Every unit we sell is supported for a "
    "full decade because our engineers obsess over reliability and long service life "
    "above all other considerations. Customers in mining, automotive, and heavy "
    "manufacturing rely on these machines to run continuously through long and "
    "demanding production shifts every working week of the year."
)

ABOUT_CONTENT = (
    "Founded in 1990 by three engineers with a passion for precision manufacturing. "
    "Acme Corporation started in a small garage and grew into a leading supplier of "
    "industrial equipment. Our commitment to quality and innovation has made us the "
    "trusted partner for thousands of manufacturing facilities worldwide. Today we "
    "employ over two hundred skilled engineers and technicians dedicated to delivering "
    "excellence in every product we manufacture and every customer interaction we "
    "undertake."
)

PRICING_CONTENT = (
    "Our pricing model is transparent and competitive. Standard widgets start at five "
    "thousand dollars and scale based on customization and volume. We offer flexible "
    "payment terms and volume discounts for enterprise customers. Contact our sales "
    "team for a custom quote tailored to your specific production needs and budget "
    "requirements. We also provide financing options and lease programs for companies "
    "seeking flexibility in their equipment investment and cash flow management "
    "strategies."
)

CONTACT_CONTENT = (
    "Reach our customer support team at support at acme dot com or call one eight "
    "hundred acme four five six seven. Our sales representatives are available Monday "
    "through Friday from eight in the morning until five in the evening Eastern "
    "Standard Time. We maintain offices in New York, San Francisco, and Frankfurt to "
    "serve customers across multiple time zones and regions. Visit our headquarters "
    "at one two three industrial way Springfield Illinois for product demonstrations "
    "and technical consultations."
)

HOME_HTML = (
    "<html><head><title>Acme Co</title>"
    '<meta property="og:site_name" content="Acme Corporation"></head>'
    f"<body><nav>home about pricing contact</nav><main>{HOME_CONTENT}</main>"
    '<a href="/about">About</a><a href="/pricing">Pricing</a><a href="/contact">Contact</a>'
    "<footer>copyright acme 1990</footer></body></html>"
)
ABOUT_HTML = (
    "<html><head><title>About Us</title></head>"
    f"<body><main>{ABOUT_CONTENT}</main></body></html>"
)
PRICING_HTML = (
    "<html><head><title>Pricing & Plans</title></head>"
    f"<body><main>{PRICING_CONTENT}</main></body></html>"
)
CONTACT_HTML = (
    "<html><head><title>Contact</title></head>"
    f"<body><main>{CONTACT_CONTENT}</main></body></html>"
)


def _install_site(monkeypatch: pytest.MonkeyPatch, routes: dict[str, tuple[int, str]]) -> None:
    """Point crawl_site at a fixture site. `routes` maps URL path -> (status, body);
    unknown paths (e.g. /sitemap.xml) return 404 so the link-crawl fallback kicks in."""

    def handler(request: httpx.Request) -> httpx.Response:
        status, body = routes.get(request.url.path, (404, ""))
        return httpx.Response(status, text=body, headers={"content-type": "text/html"})

    def make_client() -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=True)

    monkeypatch.setattr(crawl_mod, "_make_client", make_client)


ALLOW_ROBOTS = (200, "User-agent: *\nAllow: /\n")


async def test_extracts_pages_without_network(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_site(
        monkeypatch,
        {
            "/robots.txt": ALLOW_ROBOTS,
            "/": (200, HOME_HTML),
            "/about": (200, ABOUT_HTML),
            "/pricing": (200, PRICING_HTML),
        },
    )

    result = await crawl_site(AuditState(run_id="t1", url=BASE))
    pages = result["pages"]

    assert len(pages) >= 3
    assert result["brand_name"] == "Acme Corporation"
    # boilerplate stripped: nav/footer text must not survive into content
    home = next(p for p in pages if p.url == BASE)
    assert "copyright" not in home.content
    assert "contact" not in home.content  # nav word
    assert result.get("status", "running") != "failed_insufficient_content"


async def test_robots_disallow_respected(monkeypatch: pytest.MonkeyPatch) -> None:
    home_with_secret = HOME_HTML.replace(
        '<a href="/pricing">Pricing</a>',
        '<a href="/pricing">Pricing</a><a href="/private/secret">Secret</a>',
    )
    _install_site(
        monkeypatch,
        {
            "/robots.txt": (200, "User-agent: *\nDisallow: /private\n"),
            "/": (200, home_with_secret),
            "/about": (200, ABOUT_HTML),
            "/pricing": (200, PRICING_HTML),
            "/private/secret": (200, f"<html><body><main>{HOME_CONTENT}</main></body></html>"),
        },
    )

    result = await crawl_site(AuditState(run_id="t2", url=BASE))

    crawled = [p.url for p in result["pages"]]
    assert crawled, "expected some pages"
    assert not any("/private" in url for url in crawled)


async def test_js_only_site_insufficient(monkeypatch: pytest.MonkeyPatch) -> None:
    js_shell = (
        '<html><head><title>SPA</title></head><body><div id="root"></div>'
        '<script>document.write("hi")</script></body></html>'
    )
    _install_site(monkeypatch, {"/robots.txt": ALLOW_ROBOTS, "/": (200, js_shell)})

    result = await crawl_site(AuditState(run_id="t3", url=BASE))

    assert result["status"] == "failed_insufficient_content"
    assert any("insufficient content" in err for err in result["errors"])
