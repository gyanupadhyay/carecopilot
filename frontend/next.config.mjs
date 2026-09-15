/** @type {import('next').NextConfig} */
const nextConfig = {
  // Emits a self-contained server bundle with only the node_modules actually
  // imported, so the runtime Docker stage copies ~50MB instead of the whole
  // dependency tree. See docker/frontend.Dockerfile, which sets DOCKER_BUILD.
  //
  // Conditional because the standalone server is what a container runs, and
  // is the wrong artifact everywhere else: a host that deploys Next itself
  // (Vercel) builds its own output, and `standalone` there is at best
  // redundant and at worst a second server bundle nothing serves.
  output: process.env.DOCKER_BUILD ? "standalone" : undefined,
  reactStrictMode: true,
  // The API is a separate origin (FastAPI on :8000), reached directly from
  // the browser with CORS rather than proxied through Next. A proxy would
  // put the SSE stream through an extra hop that buffers it.
  env: {
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000",
  },
};

export default nextConfig;
