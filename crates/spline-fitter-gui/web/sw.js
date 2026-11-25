// Offline use of the browser version (registered by index.html).
//
// Network first: online, every file is fetched fresh (revalidated, so a new
// release shows at the next load and the app's files always match each
// other); each answer is kept, and offline the kept copies serve the app.
// The app's files are kept at install, so it works offline after one visit.
const CACHE = "airfoil-spline-fitter";
const FILES = [
  "./",
  "index.html",
  "spline-fitter.js",
  "spline-fitter_bg.wasm",
  "fit-worker.js",
  "fit-worker_bg.wasm",
  "fit-worker_loader.js",
  "fs.js",
  "manifest.webmanifest",
  "app_icon.png",
];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE).then((cache) => cache.addAll(FILES.map((f) => new Request(f, { cache: "no-cache" })))).catch(() => {}),
  );
  self.skipWaiting();
});

self.addEventListener("activate", (event) => event.waitUntil(self.clients.claim()));

self.addEventListener("fetch", (event) => {
  const request = event.request;
  if (request.method !== "GET" || new URL(request.url).origin !== self.location.origin) return;
  event.respondWith(
    fetch(request, { cache: "no-cache" })
      .then((response) => {
        if (response.ok) {
          const copy = response.clone();
          caches.open(CACHE).then((cache) => cache.put(request, copy));
        }
        return response;
      })
      .catch(() => caches.match(request, { ignoreSearch: true }).then((hit) => hit || Response.error())),
  );
});
