import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: './e2e',
  timeout: 10 * 60 * 1000,
  expect: {
    timeout: 30 * 1000,
  },
  use: {
    baseURL: process.env.PLAYWRIGHT_BASE_URL || 'https://d2mwxp9ivx7w3g.cloudfront.net',
    headless: true,
    trace: 'retain-on-failure',
  },
  workers: 1,
  reporter: [['list']],
});
