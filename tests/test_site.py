"""Check the small Wallace introduction page's cross-file contracts."""

import unittest
from html.parser import HTMLParser
from pathlib import Path
from xml.etree import ElementTree


SITE = Path(__file__).resolve().parents[1] / "test_site"


class PageParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.elements = []

    def handle_starttag(self, tag, attrs):
        self.elements.append((tag, dict(attrs)))


class SiteTests(unittest.TestCase):
    def test_page_links_existing_build_assets_and_real_repository(self):
        parser = PageParser()
        parser.feed((SITE / "index.html").read_text(encoding="utf-8"))
        assets = [
            attrs[attribute]
            for tag, attrs in parser.elements
            for attribute in ("src", "href")
            if attribute in attrs and (tag == "script" or tag == "link")
        ]
        self.assertIn("./styles.css", assets)
        self.assertIn("./app.js", assets)
        self.assertIn("./favicon.svg", assets)
        self.assertNotIn("./app.ts", assets)
        for asset in assets:
            self.assertTrue((SITE / asset).is_file(), f"Missing page asset: {asset}")
        links = [attrs.get("href", "") for tag, attrs in parser.elements if tag == "a"]
        self.assertIn("https://github.com/joshwmerrell/langchain-starter-agent", links)
        self.assertFalse(any("your-repo" in link for link in links))

    def test_interactive_elements_match_the_script_contract(self):
        parser = PageParser()
        parser.feed((SITE / "index.html").read_text(encoding="utf-8"))
        ids = {attrs["id"] for _, attrs in parser.elements if "id" in attrs}
        self.assertTrue({"theme-toggle", "detail-title", "detail-description"} <= ids)
        features = {attrs["data-feature"] for _, attrs in parser.elements if "data-feature" in attrs}
        self.assertEqual(features, {"creative", "practical", "productivity", "development"})
        ElementTree.parse(SITE / "favicon.svg")


if __name__ == "__main__":
    unittest.main()
