import { test, expect, chromium } from '@playwright/test';

test('ManualControl init load cancels cleanly across navigation', async () => {
    const browser = await chromium.launch({ channel: 'msedge', headless: true });
    const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
    const consoleMessages = [];
    const pageErrors = [];
    const streamRequests = [];

    page.on('console', (msg) => {
        consoleMessages.push({ type: msg.type(), text: msg.text() });
    });

    page.on('pageerror', (error) => {
        pageErrors.push(String(error));
    });

    page.on('request', (request) => {
        if (request.url().includes('/api/cameras/stream/')) {
            streamRequests.push(request.url());
        }
    });

    await page.route('**/api/manualcontrol/motors', async (route) => {
        await new Promise((resolve) => setTimeout(resolve, 1500));
        await route.continue();
    });

    await page.route('**/api/cameras/list', async (route) => {
        await new Promise((resolve) => setTimeout(resolve, 1500));
        await route.continue();
    });

    try {
        await page.goto('http://127.0.0.1/manual-control', {
            waitUntil: 'domcontentloaded',
            timeout: 30000,
        });
        await page.waitForTimeout(100);

        await page.goto('http://127.0.0.1/about', {
            waitUntil: 'domcontentloaded',
            timeout: 30000,
        });
        await page.waitForTimeout(2300);

        await page.goto('http://127.0.0.1/manual-control', {
            waitUntil: 'domcontentloaded',
            timeout: 30000,
        });
        await page.waitForTimeout(5000);

        const setupVisible = await page.getByTitle('Setup').isVisible().catch(() => false);
        const loadingVisible = await page.getByText('Loading...').isVisible().catch(() => false);
        const relevantConsole = consoleMessages.filter((entry) => {
            const text = entry.text.toLowerCase();
            return (
                text.includes('manual control')
                || text.includes('manualcontrol')
                || text.includes('reticle')
                || text.includes('state update')
                || text.includes('unmount')
                || text.includes('abort')
            );
        });
        const bodyText = ((await page.locator('body').textContent()) || '').replace(/\s+/g, ' ').trim();

        console.log(JSON.stringify({
            finalUrl: page.url(),
            setupVisible,
            loadingVisible,
            streamRequestCount: streamRequests.length,
            lastStreamRequest: streamRequests[streamRequests.length - 1] || null,
            relevantConsole,
            pageErrors,
            bodySnippet: bodyText.slice(0, 500),
        }, null, 2));

        expect(page.url()).toContain('/manual-control');
        expect(setupVisible).toBe(true);
        expect(loadingVisible).toBe(false);
        expect(streamRequests.length).toBeGreaterThan(0);
        expect(relevantConsole).toEqual([]);
        expect(pageErrors).toEqual([]);
    } finally {
        await browser.close();
    }
});
