// Minimal service worker for PWA installation
// This enables the "Install" prompt in Chrome

self.addEventListener('install', (event) => {
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil(clients.claim());
});

self.addEventListener('fetch', (event) => {
  // Pass through all requests (no caching)
  event.respondWith(fetch(event.request));
});
