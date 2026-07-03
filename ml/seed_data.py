"""
Seed sample analytics data for testing all ML models.
Generates realistic-looking session, pageview, event, heatmap data.
"""

import os, sys, json, uuid, random, math
from datetime import datetime, timedelta
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.resolve()))

from ml.config import CONFIG
import psycopg2
import psycopg2.extras

random.seed(42)

PAGES = [
    '/', '/about', '/pricing', '/signup', '/login',
    '/products', '/products/category-1', '/products/category-2',
    '/product/100', '/product/200', '/product/300', '/product/400',
    '/blog', '/blog/post-1', '/blog/post-2', '/blog/post-3',
    '/cart', '/checkout', '/thank-you', '/contact',
    '/support', '/faq', '/docs', '/docs/getting-started',
    '/search', '/dashboard', '/settings', '/profile',
]

REFERRERS = ['google.com', 'facebook.com', 'twitter.com', 'direct', 'bing.com', 'linkedin.com', 'github.io']
BROWSERS = ['Chrome', 'Firefox', 'Safari', 'Edge']
OSES = ['Windows', 'macOS', 'Linux', 'iOS', 'Android']
DEVICES = ['desktop', 'mobile', 'tablet']
COUNTRIES = ['US', 'GB', 'DE', 'FR', 'JP', 'CA', 'AU', 'BR', 'IN', 'SG']
CITIES = ['New York', 'London', 'Berlin', 'Paris', 'Tokyo', 'Toronto', 'Sydney', 'Sao Paulo', 'Mumbai', 'Singapore']
SCREENS = ['1920x1080', '1440x900', '1366x768', '375x812', '414x896']

SESSION_PATTERNS = [
    ['/', '/products', '/product/100', '/cart', '/checkout'],
    ['/', '/about', '/contact'],
    ['/', '/blog', '/blog/post-1'],
    ['/', '/products', '/products/category-1', '/product/200'],
    ['/', '/pricing', '/signup'],
    ['/', '/login', '/dashboard'],
    ['/', '/search', '/products', '/cart', '/checkout', '/thank-you'],
    ['/', '/blog', '/blog/post-2', '/blog'],
    ['/', '/support', '/faq'],
    ['/', '/products', '/product/300', '/cart'],
    ['/', '/docs', '/docs/getting-started'],
    ['/', '/products', '/products/category-2', '/product/400', '/cart', '/checkout'],
    ['/', '/about', '/pricing', '/signup'],
    ['/', '/blog', '/blog/post-3', '/blog'],
    ['/', '/search', '/search'],
    ['/', '/products', '/product/100', '/product/200'],
    ['/', '/contact', '/support'],
    ['/', '/pricing', '/products', '/product/300'],
    ['/', '/login', '/settings', '/profile'],
    ['/', '/products', '/cart', '/checkout', '/thank-you'],
]


def seed_data(db_url: str, days: int = 90, sessions_per_day: int = 100):
    conn = psycopg2.connect(db_url)
    cur = conn.cursor()
    
    # Create a demo website
    website_id = '41e2b680-648e-4b09-bcd7-3e2b10c06264'
    cur.execute("""
        INSERT INTO website (website_id, name, domain, created_at)
        VALUES (%s, 'Demo Site', 'example.com', NOW())
        ON CONFLICT (website_id) DO NOTHING
    """, (website_id,))
    
    # Check if admin user exists
    cur.execute("SELECT user_id FROM \"user\" LIMIT 1")
    admin = cur.fetchone()
    admin_id = admin[0] if admin else '41e2b680-648e-4b09-bcd7-3e2b10c06264'
    
    conn.commit()
    print(f"Website: {website_id}")
    
    total_events = 0
    total_sessions = 0
    
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)
    
    for day_offset in range(days):
        day_date = start_date + timedelta(days=day_offset)
        day_start = day_date.replace(hour=0, minute=0, second=0, microsecond=0)
        
        for _ in range(sessions_per_day):
            session_id = str(uuid.uuid4())
            visit_id = str(uuid.uuid4())
            
            pattern = random.choice(SESSION_PATTERNS)
            session_start = day_start + timedelta(
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59),
                seconds=random.randint(0, 59),
            )
            
            browser = random.choice(BROWSERS)
            os_name = random.choice(OSES)
            device = random.choice(DEVICES)
            country = random.choice(COUNTRIES)
            city = random.choice(CITIES)
            screen = random.choice(SCREENS)
            referrer = random.choice(REFERRERS)
            language = 'en-US' if random.random() > 0.3 else 'de-DE'
            
            # Insert session
            cur.execute("""
                INSERT INTO session (session_id, website_id, browser, os, device, screen, language, country, city, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (session_id, created_at) DO NOTHING
            """, (session_id, website_id, browser, os_name, device, screen, language, country, city, session_start))
            
            # Insert pageview events
            for step_idx, page in enumerate(pattern):
                event_time = session_start + timedelta(seconds=step_idx * random.randint(3, 30))
                event_id = str(uuid.uuid4())
                
                is_checkout = 'checkout' in page or 'cart' in page
                is_signup = 'signup' in page
                
                utm_source = '' if random.random() > 0.3 else random.choice(['google', 'facebook', 'twitter'])
                
                # Performance metrics
                lcp = round(random.uniform(0.5, 4.0), 1)
                cls = round(random.uniform(0.0, 0.5), 4)
                inp = round(random.uniform(10, 500), 1)
                fcp = round(random.uniform(0.3, 3.0), 1)
                ttfb = round(random.uniform(0.1, 2.0), 1)
                
                cur.execute("""
                    INSERT INTO website_event (event_id, website_id, session_id, visit_id,
                        url_path, referrer_domain, page_title, event_type,
                        lcp, cls, inp, fcp, ttfb, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, 1, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (event_id, created_at) DO NOTHING
                """, (event_id, website_id, session_id, visit_id,
                      page, referrer, f'Demo Page {step_idx}',
                      lcp, cls, inp, fcp, ttfb, event_time))
                
                total_events += 1
                
                # Add event_data for conversions
                if is_checkout and random.random() > 0.6:
                    data_id = str(uuid.uuid4())
                    cur.execute("""
                        INSERT INTO event_data (event_data_id, website_id, website_event_id,
                            data_key, string_value, data_type, created_at)
                        VALUES (%s, %s, %s, 'revenue', %s, 2, %s)
                        ON CONFLICT (event_data_id, created_at) DO NOTHING
                    """, (data_id, website_id, event_id, str(round(random.uniform(10, 200), 2)), event_time))
                    
                    rev_id = str(uuid.uuid4())
                    cur.execute("""
                        INSERT INTO revenue (revenue_id, website_id, session_id, event_id, event_name, currency, revenue, created_at)
                        VALUES (%s, %s, %s, %s, 'purchase', 'USD', %s, %s)
                        ON CONFLICT (revenue_id, created_at) DO NOTHING
                    """, (rev_id, website_id, session_id, event_id, round(random.uniform(10, 200), 2), event_time))
            
            # Add heatmap events (with occasional rage clicks)
            n_clicks = random.randint(0, 12)
            rage_session = random.random() < 0.05  # 5% of sessions have rage clicks
            for c in range(n_clicks):
                heat_id = str(uuid.uuid4())
                x = random.randint(50, 1200)
                y = random.randint(50, 2000)
                scroll_pct = random.randint(10, 100)
                
                # Rage click pattern: 3+ clicks on same spot within 1s
                if rage_session and c < 4:
                    x = 500 + random.randint(-5, 5)
                    y = 300 + random.randint(-5, 5)
                    scroll_pct = 30
                
                heat_time = session_start + timedelta(seconds=c * random.randint(1, 3))
                cur.execute("""
                    INSERT INTO heatmap_event (heatmap_event_id, website_id, session_id, visit_id,
                        url_path, event_type, x, y, scroll_pct,
                        viewport_w, viewport_h, created_at)
                    VALUES (%s, %s, %s, %s, %s, 1, %s, %s, %s, 1920, 1080, %s)
                    ON CONFLICT (heatmap_event_id, created_at) DO NOTHING
                """, (heat_id, website_id, session_id, visit_id,
                      pattern[0] if pattern else '/', x, y, scroll_pct, heat_time))
            
            total_sessions += 1
        
        if (day_offset + 1) % 10 == 0:
            conn.commit()
            print(f"  Day {day_offset+1}/{days}: {total_sessions} sessions, {total_events} events")
    
    conn.commit()
    cur.close()
    conn.close()
    
    print(f"\nSeeded: {total_sessions} sessions, {total_events} events across {days} days")
    return website_id


if __name__ == '__main__':
    print("Seeding umami analytics data...")
    wid = seed_data(CONFIG.db.url, days=90, sessions_per_day=100)
    print(f"Website ID: {wid}")
