/** @type {import('next').NextConfig} */
const nextConfig = {
  // Emits a self-contained server bundle with only the node_modules actually
  // imported, so the runtime Docker stage copies ~50MB instead of the whole
  // dependency tree. See docker/frontend.Dockerfile.
  output: "standalone",
  reactStrictMode: true,
  // The API is a separate origin (FastAPI on :8000), reached directly from
  // the browser with CORS rather than proxied through Next. A proxy would
  // put the SSE stream through an extra hop that buffers it.
  env: {
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000",
  },
};

export default nextConfig;
