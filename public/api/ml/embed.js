/**
 * umami.recommend() - Client-side recommendation embed API
 *
 * Drop this script on any page to get real-time content recommendations
 * powered by the umami ML service.
 *
 * Usage:
 *   <script src="https://your-umami-instance.com/api/ml/embed.js" data-website-id="xxx"></script>
 *   <div class="umami-recommendations"></div>
 *
 * The script auto-discovers <div class="umami-recommendations"> elements
 * and populates them with recommended page links.
 */

(function () {
  'use strict';

  const SCRIPT = document.querySelector('script[src*="api/ml/embed"]');
  if (!SCRIPT) return;

  const WEBSITE_ID = SCRIPT.getAttribute('data-website-id');
  const ML_API = SCRIPT.getAttribute('data-ml-api') || '/api/ml';
  const TOP_K = parseInt(SCRIPT.getAttribute('data-top-k') || '5', 10);
  const THEME = SCRIPT.getAttribute('data-theme') || 'light';

  if (!WEBSITE_ID) {
    console.warn('[umami.recommend] Missing data-website-id attribute');
    return;
  }

  // Collect session context from the page
  function getSessionContext() {
    return {
      current_page: window.location.pathname,
      referrer: document.referrer || '',
      device: /Mobi|Android/i.test(navigator.userAgent) ? 'mobile' : 'desktop',
      browser: navigator.userAgent.includes('Chrome') ? 'Chrome'
        : navigator.userAgent.includes('Firefox') ? 'Firefox'
        : navigator.userAgent.includes('Safari') ? 'Safari'
        : 'Other',
      screen: `${screen.width}x${screen.height}`,
      language: navigator.language,
      hour: new Date().getHours(),
      is_weekend: [0, 6].includes(new Date().getDay()),
      session_depth: parseInt(sessionStorage.getItem('umami_page_depth') || '1', 10),
      time_on_site: Math.floor((Date.now() - performance.timing.navigationStart) / 1000),
    };
  }

  // Track page depth
  const depth = parseInt(sessionStorage.getItem('umami_page_depth') || '0', 10) + 1;
  sessionStorage.setItem('umami_page_depth', String(depth));

  // Fetch recommendations from the ML API
  async function fetchRecommendations() {
    try {
      const response = await fetch(`${ML_API}/recommend`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          websiteId: WEBSITE_ID,
          sessionPages: [window.location.pathname],
          sessionFeatures: getSessionContext(),
          topK: TOP_K,
        }),
      });

      if (!response.ok) return [];

      const data = await response.json();
      return data.recommendations || [];
    } catch (e) {
      console.warn('[umami.recommend] API call failed:', e);
      return [];
    }
  }

  // Render recommendations into target elements
  function renderRecommendations(recommendations) {
    const containers = document.querySelectorAll('.umami-recommendations');
    if (!containers.length) return;

    const isDark = THEME === 'dark' || document.documentElement.classList.contains('dark');

    const style = document.createElement('style');
    style.textContent = `
      .umami-rec-widget {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        max-width: 100%;
      }
      .umami-rec-title {
        font-size: 14px;
        font-weight: 600;
        margin-bottom: 8px;
        color: ${isDark ? '#e5e7eb' : '#374151'};
      }
      .umami-rec-item {
        display: flex;
        align-items: center;
        padding: 8px 12px;
        margin-bottom: 4px;
        border-radius: 6px;
        text-decoration: none;
        transition: background 0.15s;
        background: ${isDark ? 'rgba(255,255,255,0.05)' : 'rgba(59,130,246,0.05)'};
      }
      .umami-rec-item:hover {
        background: ${isDark ? 'rgba(255,255,255,0.1)' : 'rgba(59,130,246,0.1)'};
      }
      .umami-rec-rank {
        font-size: 12px;
        font-weight: 700;
        color: ${isDark ? '#60a5fa' : '#2563eb'};
        min-width: 20px;
      }
      .umami-rec-page {
        font-size: 13px;
        color: ${isDark ? '#d1d5db' : '#4b5563'};
        flex: 1;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
      }
      .umami-rec-score {
        font-size: 11px;
        color: ${isDark ? '#9ca3af' : '#9ca3af'};
        margin-left: 8px;
      }
    `;
    document.head.appendChild(style);

    containers.forEach(container => {
      container.classList.add('umami-rec-widget');

      const title = document.createElement('div');
      title.className = 'umami-rec-title';
      title.textContent = 'Recommended';
      container.appendChild(title);

      if (!recommendations.length) {
        const empty = document.createElement('div');
        empty.className = 'umami-rec-item';
        empty.textContent = 'No recommendations yet';
        container.appendChild(empty);
        return;
      }

      recommendations.forEach((rec, i) => {
        const link = document.createElement('a');
        link.className = 'umami-rec-item';
        link.href = rec.page;
        link.innerHTML = `
          <span class="umami-rec-rank">${i + 1}</span>
          <span class="umami-rec-page">${rec.page}</span>
          <span class="umami-rec-score">${(rec.score * 100).toFixed(0)}%</span>
        `;

        // Track click
        link.addEventListener('click', () => {
          try {
            navigator.sendBeacon(`${ML_API}/recommend/click`, JSON.stringify({
              websiteId: WEBSITE_ID,
              page: rec.page,
              score: rec.score,
            }));
          } catch (e) { /* ignore */ }
        });

        container.appendChild(link);
      });
    });
  }

  // Initialize
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', async () => {
      const recs = await fetchRecommendations();
      renderRecommendations(recs);
    });
  } else {
    fetchRecommendations().then(renderRecommendations);
  }
})();
