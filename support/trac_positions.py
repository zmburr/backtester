"""
TRAC Position Scraper — Scrapes open positions from trac.trilliumtrading.com

Provides a singleton browser session that can be shared across modules.
Designed to be imported by bounce_trader.py for position-aware alerts.

Usage:
    from support.trac_positions import TracPositionScraper

    # Initialize once (opens browser for manual login)
    scraper = TracPositionScraper()
    scraper.login()  # Prompts for manual login

    # Get current positions (can be called repeatedly)
    positions = scraper.get_positions()
    # Returns: [{'symbol': 'NVDA', 'shares': 500, 'side': 'long', 'avg_price': 120.50, 'bp_used': 60250.0}, ...]

    # Check if a specific ticker is in positions
    if scraper.has_position('NVDA'):
        pos = scraper.get_position('NVDA')
        print(f"Long {pos['shares']} shares @ {pos['avg_price']}")
"""

import logging
import threading
from typing import Dict, List, Optional
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('TracPositions')

# URLs
TRAC_URL = 'https://trac.trilliumtrading.com/'
LOGIN_URL = TRAC_URL + 'Login'
DAILY_SUMMARY_URL = TRAC_URL + 'dailysummary.action'


def parse_currency(td_element) -> float:
    """Parse currency value from table cell. Handles $820.30, -$550.19, ($225.13)."""
    div = td_element.find('div')
    text = div.get_text(strip=True) if div else td_element.get_text(strip=True)
    text = text.replace("$", "").replace(",", "")
    is_negative = text.startswith("-") or text.startswith("(")
    text = text.replace("-", "").replace("(", "").replace(")", "")
    try:
        value = float(text)
        return -value if is_negative else value
    except ValueError:
        return 0.0


def scrape_positions_from_html(html: str) -> List[Dict]:
    """
    Parse the dailyOpenPosTbl table from Daily Summary HTML.

    Returns list of dicts with: symbol, shares, side, avg_price, bp_used
    """
    positions = []
    try:
        soup = BeautifulSoup(html, "html.parser")
        table = soup.find('table', {'id': 'dailyOpenPosTbl'})
        if not table:
            logger.debug("Could not find dailyOpenPosTbl")
            return positions

        tbody = table.find('tbody')
        if not tbody:
            return positions

        rows = tbody.find_all('tr')
        for row in rows:
            tds = row.find_all('td')
            if not tds or len(tds) < 3:
                continue

            full_symbol_str = tds[0].get_text(strip=True)
            # Skip options
            if "Call" in full_symbol_str or "Put" in full_symbol_str:
                continue

            underlying = full_symbol_str.split()[0]
            pos_str = tds[2].get_text(strip=True).replace(",", "")
            try:
                current_pos = int(pos_str)
            except ValueError:
                continue

            if abs(current_pos) < 10:
                continue

            side = "long" if current_pos > 0 else "short"

            # Extract average price from column 5
            avg_price = None
            try:
                if len(tds) > 5:
                    div_element = tds[5].find('div')
                    if div_element:
                        avg_price_str = div_element.get_text(strip=True).replace("$", "").replace(",", "")
                    else:
                        avg_price_str = tds[5].get_text(strip=True).replace("$", "").replace(",", "")
                    if avg_price_str:
                        avg_price = float(avg_price_str)
            except (ValueError, IndexError):
                pass

            # Extract BP used from column 8
            bp_used = None
            try:
                if len(tds) > 8:
                    div_element = tds[8].find('div')
                    if div_element:
                        bp_used_str = div_element.get_text(strip=True).replace("$", "").replace(",", "")
                    else:
                        bp_used_str = tds[8].get_text(strip=True).replace("$", "").replace(",", "")
                    if bp_used_str:
                        bp_used = float(bp_used_str)
            except (ValueError, IndexError):
                pass

            positions.append({
                "symbol": underlying,
                "shares": abs(current_pos),
                "side": side,
                "avg_price": avg_price,
                "bp_used": bp_used,
            })
    except Exception as e:
        logger.warning(f"Exception scraping positions: {e}")
    return positions


class TracPositionScraper:
    """
    Singleton-style scraper for TRAC positions.

    Opens a Selenium browser for manual login, then provides methods
    to fetch current open positions on demand.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.driver = None
        self.logged_in = False
        self._positions_cache: List[Dict] = []
        self._cache_lock = threading.Lock()
        logger.info("TracPositionScraper initialized")

    def login(self, headless: bool = False):
        """
        Open browser and prompt for manual login.

        Args:
            headless: If True, run browser in headless mode (not recommended for login)
        """
        if self.logged_in and self.driver:
            logger.info("Already logged in")
            return

        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options

            options = Options()
            if headless:
                options.add_argument('--headless')

            self.driver = webdriver.Chrome(options=options)
            self.driver.get(LOGIN_URL)
            logger.info("Browser opened at TRAC login page. Please log in manually.")
            input("After logging in, press Enter to continue...")

            # Navigate to Daily Summary
            self.driver.get(DAILY_SUMMARY_URL)
            self.logged_in = True
            logger.info("Logged in and navigated to Daily Summary")

        except Exception as e:
            logger.error(f"Failed to initialize browser: {e}")
            raise

    def refresh(self):
        """Refresh the Daily Summary page."""
        if not self.driver or not self.logged_in:
            logger.warning("Not logged in — call login() first")
            return
        try:
            self.driver.refresh()
        except Exception as e:
            logger.warning(f"Failed to refresh: {e}")

    def get_positions(self, refresh: bool = True) -> List[Dict]:
        """
        Get current open positions.

        Args:
            refresh: If True, refresh the page before scraping

        Returns:
            List of position dicts with keys: symbol, shares, side, avg_price, bp_used
        """
        if not self.driver or not self.logged_in:
            logger.warning("Not logged in — returning cached positions")
            return self._positions_cache

        try:
            if refresh:
                self.driver.refresh()
                import time
                time.sleep(1)  # Brief wait for page load

            html = self.driver.page_source
            positions = scrape_positions_from_html(html)

            with self._cache_lock:
                self._positions_cache = positions

            return positions

        except Exception as e:
            logger.warning(f"Failed to get positions: {e}")
            return self._positions_cache

    def has_position(self, symbol: str, refresh: bool = False) -> bool:
        """Check if a position exists for the given symbol."""
        positions = self.get_positions(refresh=refresh)
        return any(p['symbol'].upper() == symbol.upper() for p in positions)

    def get_position(self, symbol: str, refresh: bool = False) -> Optional[Dict]:
        """Get position data for a specific symbol, or None if not found."""
        positions = self.get_positions(refresh=refresh)
        for p in positions:
            if p['symbol'].upper() == symbol.upper():
                return p
        return None

    def get_position_symbols(self, refresh: bool = False) -> List[str]:
        """Get list of all symbols with open positions."""
        positions = self.get_positions(refresh=refresh)
        return [p['symbol'] for p in positions]

    def close(self):
        """Close the browser."""
        if self.driver:
            try:
                self.driver.quit()
            except Exception:
                pass
            self.driver = None
            self.logged_in = False
            logger.info("Browser closed")


# Module-level convenience functions
_scraper: Optional[TracPositionScraper] = None


def get_scraper() -> TracPositionScraper:
    """Get or create the singleton scraper instance."""
    global _scraper
    if _scraper is None:
        _scraper = TracPositionScraper()
    return _scraper


def get_open_positions(refresh: bool = True) -> List[Dict]:
    """Get current open positions (requires login() to have been called)."""
    return get_scraper().get_positions(refresh=refresh)


def has_position(symbol: str) -> bool:
    """Check if we have an open position in the given symbol."""
    return get_scraper().has_position(symbol, refresh=False)


def get_position(symbol: str) -> Optional[Dict]:
    """Get position data for a symbol, or None if no position."""
    return get_scraper().get_position(symbol, refresh=False)


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("=== TRAC Position Scraper Test ===")
    scraper = TracPositionScraper()
    scraper.login()

    print("\nFetching positions...")
    positions = scraper.get_positions()

    if positions:
        print(f"\nFound {len(positions)} open positions:")
        for p in positions:
            avg = p.get('avg_price')
            avg_str = f"{avg:.2f}" if avg is not None else "N/A"
            print(f"  {p['symbol']}: {p['side']} {p['shares']} shares @ ${avg_str}")
    else:
        print("\nNo open positions found.")

    # Test lookup
    test_symbol = input("\nEnter a symbol to check (or press Enter to skip): ").strip().upper()
    if test_symbol:
        if scraper.has_position(test_symbol):
            pos = scraper.get_position(test_symbol)
            print(f"  Position found: {pos}")
        else:
            print(f"  No position in {test_symbol}")

    scraper.close()
    print("\nDone.")
