# CareCopilot web UI.
#
# Three stages. The runtime image carries Next's standalone output only —
# no npm, no source, no dev dependencies, and only the node_modules the
# build actually imported.

# --- dependencies --------------------------------------------------------- #
FROM node:24-alpine AS deps

WORKDIR /app
COPY frontend/package.json frontend/package-lock.json* ./
# `npm ci` when a lockfile is present (reproducible), `npm install` when it
# is not, so a fresh clone without one still builds.
RUN if [ -f package-lock.json ]; then npm ci; else npm install; fi

# --- build ---------------------------------------------------------------- #
FROM node:24-alpine AS build

WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY frontend/ ./

# Next inlines NEXT_PUBLIC_* into the client bundle at build time, so this
# has to be an ARG rather than a runtime environment variable. It is the URL
# the browser calls, which is why it is a published host and not the compose
# service name — the browser cannot resolve "backend".
ARG NEXT_PUBLIC_API_URL=http://localhost:8000
ENV NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL
ENV NEXT_TELEMETRY_DISABLED=1

RUN npm run build

# --- runtime -------------------------------------------------------------- #
FROM node:24-alpine AS runtime

ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1 \
    PORT=3000 \
    HOSTNAME=0.0.0.0

WORKDIR /app
RUN addgroup -g 10003 -S web && adduser -u 10003 -S web -G web

# The standalone bundle is the server plus its traced dependencies; static
# assets are not traced into it and have to be copied alongside.
COPY --from=build --chown=web:web /app/.next/standalone ./
COPY --from=build --chown=web:web /app/.next/static ./.next/static
COPY --from=build --chown=web:web /app/public ./public

USER web
EXPOSE 3000

HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=3 \
    CMD node -e "fetch('http://localhost:3000').then(r=>process.exit(r.ok?0:1)).catch(()=>process.exit(1))"

CMD ["node", "server.js"]
