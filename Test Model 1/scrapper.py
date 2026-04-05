import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from playwright.sync_api import sync_playwright


class InstagramCommentsScraper:
    """Simple Instagram scraper that only exports raw comments."""

    def __init__(self, username: str, password: str):
        self.username = username
        self.password = password
        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None

    def start(self) -> None:
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(headless=False)
        self.context = self.browser.new_context(viewport={"width": 1280, "height": 720})
        self.page = self.context.new_page()

    def login(self) -> bool:
        self.page.goto("https://www.instagram.com/accounts/login/", wait_until="networkidle")
        self.page.wait_for_timeout(2500)

        user_input = self.page.locator("input[name='username']")
        pass_input = self.page.locator("input[name='password']")

        if user_input.count() == 0 or pass_input.count() == 0:
            return False

        user_input.fill(self.username)
        pass_input.fill(self.password)

        self.page.keyboard.press("Enter")
        self.page.wait_for_timeout(7000)

        # If login redirects away from /accounts/login, treat as success.
        return "accounts/login" not in self.page.url

    def _close_popups(self) -> None:
        candidates = ["Not Now", "Nanti Saja", "Sekarang Tidak"]
        for text in candidates:
            button = self.page.locator(f"button:has-text('{text}')")
            if button.count() > 0:
                try:
                    button.first.click(timeout=1200)
                    self.page.wait_for_timeout(1000)
                except Exception:
                    pass

    def _scroll_comments(self, max_scrolls: int = 50) -> None:
        for _ in range(max_scrolls):
            self.page.mouse.wheel(0, 4000)
            self.page.wait_for_timeout(700)

    def scrape_post_comments(self, post_url: str, max_comments: int = 1000) -> dict:
        self.page.goto(post_url, wait_until="networkidle")
        self.page.wait_for_timeout(2500)
        self._close_popups()
        self._scroll_comments()

        # Caption
        caption = "N/A"
        caption_candidates = self.page.locator("article h1, article ul li div > span")
        if caption_candidates.count() > 0:
            try:
                caption = caption_candidates.first.inner_text().strip()
            except Exception:
                caption = "N/A"

        # Comment extraction
        comments = []
        items = self.page.locator("article ul li")
        total = min(items.count(), max_comments)

        for i in range(total):
            item = items.nth(i)
            user_nodes = item.locator("a")
            text_nodes = item.locator("span")
            if user_nodes.count() == 0 or text_nodes.count() == 0:
                continue

            try:
                username = user_nodes.first.inner_text().strip()
                text = text_nodes.last.inner_text().strip()
            except Exception:
                continue

            if text:
                comments.append({"username": username, "text": text})

        owner = ""
        owner_node = self.page.locator("header a")
        if owner_node.count() > 0:
            try:
                owner = owner_node.first.inner_text().strip()
            except Exception:
                owner = ""

        return {
            "username": owner,
            "url": post_url,
            "likes": 0,
            "hearts": 0,
            "comments": len(comments),
            "comments_details": comments,
            "post_caption": caption,
        }

    def scrape_posts_comments(self, post_urls: list[str], max_comments_per_post: int = 1000) -> list[dict]:
        data = []
        for idx, url in enumerate(post_urls, start=1):
            print(f"[{idx}/{len(post_urls)}] Scraping {url}")
            try:
                post_data = self.scrape_post_comments(url, max_comments=max_comments_per_post)
                data.append(post_data)
            except Exception as exc:
                print(f"Failed for {url}: {exc}")
            time.sleep(2)
        return data

    def close(self) -> None:
        if self.context:
            self.context.close()
        if self.browser:
            self.browser.close()
        if self.playwright:
            self.playwright.stop()


def flatten_raw_comments(scraped_data: list[dict], default_topic: str) -> pd.DataFrame:
    rows = []
    for post in scraped_data:
        for comment in post.get("comments_details", []):
            rows.append(
                {
                    "TOPIK": default_topic,
                    "Username Posting Owner": post.get("username", ""),
                    "Link post IG": post.get("url", ""),
                    "Username komentar": comment.get("username", ""),
                    "Isi komentar": comment.get("text", ""),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape Instagram comments and export raw outputs.")
    parser.add_argument("--username", required=True, help="Instagram username")
    parser.add_argument("--password", required=True, help="Instagram password")
    parser.add_argument("--post-urls", nargs="+", required=True, help="One or more Instagram post URLs")
    parser.add_argument("--topic", default="Topik Umum", help="Default topic for exported flat table")
    parser.add_argument("--max-comments", type=int, default=1000, help="Maximum comments per post")
    args = parser.parse_args()

    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)

    scraper = InstagramCommentsScraper(args.username, args.password)

    try:
        scraper.start()
        if not scraper.login():
            raise RuntimeError("Login failed. Check credentials or challenge/2FA status.")

        raw_data = scraper.scrape_posts_comments(args.post_urls, max_comments_per_post=args.max_comments)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        raw_json_path = output_dir / f"raw_comments_{timestamp}.json"
        raw_csv_path = output_dir / f"raw_comments_{timestamp}.csv"

        with open(raw_json_path, "w", encoding="utf-8") as f:
            json.dump(raw_data, f, indent=2, ensure_ascii=False)

        flat_df = flatten_raw_comments(raw_data, args.topic)
        flat_df.to_csv(raw_csv_path, index=False, encoding="utf-8-sig")

        print(f"Saved raw JSON: {raw_json_path}")
        print(f"Saved raw CSV : {raw_csv_path}")
        print(f"Total comments: {len(flat_df)}")
    finally:
        scraper.close()


if __name__ == "__main__":
    main()
