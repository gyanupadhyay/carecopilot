# Deploying CareCopilot

One always-free ARM VM running the compose stack, with Caddy terminating TLS
in front of it. No managed database, no platform-as-a-service, nothing that
sleeps or auto-pauses.

The deployed stack is the stack you already run locally. That is the point of
this shape: `docker compose up` is the deployment, so there is no second
architecture to keep in step with the first, and a bug reproduced on the VM
reproduces on a laptop.

```
                    :80 :443
                       │
                  ┌────▼────┐
   internet ──────│  caddy  │  TLS, one origin
                  └────┬────┘
        ┌──────────────┼──────────────┐
        │              │              │
   /  ┌─▼────────┐  /api/* ┌──────▼──┐  /mcp ┌────▼───┐
      │ frontend │         │ backend │       │  mcp   │
      └──────────┘         └────┬────┘       └────────┘
                                │
                        ┌───────┴───────┐
                        │               │
                   ┌────▼───┐      ┌────▼───┐
                   │   db   │      │ neo4j  │   127.0.0.1 only
                   └────────┘      └────────┘
```

Caddy is the only service with a port on a public interface. Everything else
binds `127.0.0.1` — see `BIND_ADDRESS` in `.env.example` for why that default
matters more on a cloud VM than it looks.

## What changes from a local run

**Inference is hosted by default.** `LLM_PROVIDER=ollama_cloud`. The machine
has the RAM to serve a model itself — `COMPOSE_PROFILES=local-llm` brings up
`ollama` and `ollama-pull` and nothing else changes — but four ARM cores
without a GPU answer in tens of seconds, which is a demonstration rather than
a demo. Hosted is the default; local is the switch that proves the
self-hosted path still works.

Whichever you run, say which model is behind the live demo. A visitor who
assumes Qwen3-8B because the README describes it has been misled by omission.

**`ENVIRONMENT=production` turns off `/docs` and `/openapi.json`**
(`app/main.py`). The same flag gates the startup checks in `app/config.py`
that refuse to boot without `JWT_SECRET`, `ACTION_TOKEN_SECRET` and
`ANALYTICS_DATABASE_URL`. Do not switch to `staging` to get the docs back —
that silently drops all three checks. If you want public docs, make it its
own flag.

**The demo is rate-limited** (`app/api/rate_limit.py`): per-visitor limits so
one caller cannot hammer it, and a global daily budget because a per-visitor
limit does nothing against many visitors.

## Phase 1 — The machine

Oracle Cloud, 100 GB boot volume, and a shape with at least ~8 GB of RAM.
The stack sits around 3 GB at rest — Neo4j ~1.2 GB, backend ~1 GB, Postgres
~400 MB, frontend ~200 MB — with the Next.js build as a brief ~2 GB peak.

**What the free tier promises and what it delivers are different things.**
Always Free includes `VM.Standard.A1.Flex` at 4 OCPUs / 24 GB, which is the
shape to want. It is also chronically unavailable: this deployment failed on
it repeatedly in `ap-hyderabad-1` at 4/24, 2/12 and 1/6, and the paid AMD
`E4.Flex` was out of capacity too. `VM.Standard.E5.Flex` at 1 OCPU / 12 GB
was the first to launch.

So budget for capacity hunting, and know the escape routes:

- **Different shape families sit on different host pools.** Try E5, then
  Intel `VM.Standard3.Flex`, then E3. This is what eventually worked.
- **Smaller requests fit in gaps a 4-OCPU block cannot.** Step down before
  giving up on a family.
- **A different availability domain** — only if your region has more than
  one. Hyderabad does not.
- **A different region is not an option** for Always Free, which is pinned
  to your permanent home region; and trial accounts cannot subscribe to
  additional regions at all.

Anything other than A1 is a **paid** shape. During the 30-day trial it is
covered by the $300 credit (E5 at 1/12 is about $32/month, so credits are
not the constraint). **When the trial ends the account converts to Always
Free and paid instances are stopped and reclaimed** — not billed, stopped.
Put a calendar reminder a few days before. Either upgrade to Pay As You Go,
or use the window to keep retrying for an A1 and rebuild on it, which is a
`git clone` and a `docker compose up` away.

### Oracle Linux, not Ubuntu

The console resets the image whenever you change shape family, so an
instance created after a few retries is likely running **Oracle Linux**
(username `opc`) rather than Ubuntu (`ubuntu`). That is fine — everything
runs in containers — but it changes the setup commands below. Check the
**Username** field on the instance details page to see which you have.

### Two firewalls, and the second one is invisible

Traffic is filtered in two independent places. Opening one while the other is
shut produces a site that times out with no error in any log.

**Cloud:** Networking → VCN → Subnets → Security Lists → Default → add
ingress rules for TCP `80` and TCP `443` from `0.0.0.0/0`.

**Host:** Oracle's images ship a host firewall that drops everything but SSH.

Oracle Linux (`firewalld`):

```bash
sudo firewall-cmd --permanent --add-port=80/tcp
sudo firewall-cmd --permanent --add-port=443/tcp
sudo firewall-cmd --reload
```

Ubuntu (`iptables`):

```bash
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 80 -j ACCEPT
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 443 -j ACCEPT
sudo netfilter-persistent save
```

Open nothing else. Postgres and Neo4j are not meant to be reachable, and
they are not — but see the warning under `BIND_ADDRESS`: Docker publishes
ports by writing iptables rules that bypass most host firewalls, so a port
published to `0.0.0.0` is exposed *even when the firewall says otherwise*.
The base compose file binds them to loopback for exactly that reason.

### Docker

Oracle Linux — the `get.docker.com` convenience script is the Debian path and
does the wrong thing here:

```bash
sudo dnf install -y dnf-plugins-core git
sudo dnf config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo
sudo dnf install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo systemctl enable --now docker
sudo usermod -aG docker opc
exit                    # group membership needs a fresh login
```

Ubuntu:

```bash
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker ubuntu
exit
```

Log back in and check `docker run --rm hello-world` works *without* `sudo` —
if it does not, the group has not taken effect and you logged back into the
same session.

**On ARM**, should you get an A1: everything in the stack has an arm64 build,
including the one that could have stopped it cold. `onnxruntime` ships a
`cp314` manylinux aarch64 wheel, so nothing compiles from source.

## Phase 2 — A hostname

Caddy gets certificates from Let's Encrypt, which proves you control a
**name**, not an address. So a DNS record has to point at the VM before the
stack will serve HTTPS, and browsers will not let a page served over HTTPS
call an API over HTTP — there is no half-way configuration that works.

Any of these is fine:

- a domain you own → an `A` record at the VM's public IP
- a free subdomain (DuckDNS and similar) → same
- `<ip-with-dashes>.sslip.io`, which resolves to the address encoded in it
  and needs no account at all — `129.225.110.115` becomes
  `129-225-110-115.sslip.io`

Confirm it before building, because a failed challenge burns Let's Encrypt
rate limit:

```bash
getent hosts <your-hostname>     # must print the VM's public IP
curl -s https://api.ipify.org    # run on the VM; must be the same address
```

Put it in `.env` as `CARECOPILOT_DOMAIN`. `docker-compose.prod.yml` refuses
to start without it rather than failing later inside certificate issuance.

## Phase 3 — Bring it up

```bash
git clone https://github.com/gyanupadhyay/carecopilot.git
cd carecopilot
cp .env.example .env
```

Edit `.env`:

```bash
CARECOPILOT_DOMAIN=carecopilot.example.com
ENVIRONMENT=production
COMPOSE_PROFILES=                      # empty: no local model server

LLM_PROVIDER=ollama_cloud
LLM_API_KEY=...
LLM_MODEL=gemma4:31b

RERANKER=heuristic                     # one fewer model call per RAG turn

POSTGRES_PASSWORD=...                  # generate; do not keep the default
NEO4J_PASSWORD=...                     # generate
JWT_SECRET=...                         # generate
ACTION_TOKEN_SECRET=...                # generate

ANALYTICS_PASSWORD=carecopilot_ro      # pinned — see below
```

Generate each with:

```bash
python3 -c "import secrets;print(secrets.token_urlsafe(48))"
```

**`ANALYTICS_PASSWORD` cannot be randomised yet**, and it is worth knowing
why rather than discovering it as an authentication failure.
`docker/initdb/20-roles.sql` hardcodes the role's password, so changing the
variable only changes the connection string and the two stop matching.

Living with a known password here is defensible but is not nothing: the role
is reachable only from inside the compose network (the database binds
loopback), it holds SELECT on four tables, its transactions are read-only,
and row-level security it cannot bypass applies. So it is a
defence-in-depth gap rather than an exposure. The fix is to make that init
script read the password from the environment; until then, treat this as a
known follow-up.

Then:

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build
```

The first build takes a while on four ARM cores. Nothing else is needed:
`init` migrates, seeds and embeds, and `init-kg` projects into Neo4j, both
before the API accepts traffic. That is the same guarantee they give
locally — a backend that is up is one whose schema, index and graph match
its code.

Watch it come up with `docker compose logs -f`.

## Verifying

In order, because each failure points somewhere different:

1. `curl https://<domain>/api/health` → `database: ok`, `vector_backend: pgvector`
2. `/docs` returns **404** and `/api/auth/demo-accounts` returns **404** —
   both are correct in production. The second is deliberate
   (`app/api/routes/auth.py`): an endpoint that hands out working
   credentials should not be reachable in a deployment just because the data
   behind it is synthetic. It does mean a visitor cannot discover the login
   from the API, so publish it on the page or in the README.
3. Log in through the UI — TLS and the baked-in origin are right
4. "when is my next appointment?" — API route, model reachable
5. "why was I prescribed metformin?" — Neo4j reachable and populated
6. "how many lab tests have I had this year?" — the `carecopilot_ro` role
   and its RLS policies exist
7. Ask seven questions inside a minute — the seventh returns 429 with
   `Retry-After`

Then confirm the things that should *not* work, from your laptop:

```bash
nc -vz <ip> 5433     # Postgres    — must refuse
nc -vz <ip> 7687     # Neo4j bolt  — must refuse
nc -vz <ip> 7474     # Neo4j HTTP  — must refuse
```

A connection that succeeds means something is publishing to `0.0.0.0`.
Check `BIND_ADDRESS` before anything else reaches that machine.

## Updating

```bash
cd carecopilot && git pull
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build
```

`init` and `init-kg` re-run and are idempotent: `--if-empty` will not reseed
a populated database, `--only-missing` embeds only new documents, and the
graph is rebuilt from PostgreSQL. Conversations and bookings survive.

Changing `CARECOPILOT_DOMAIN` needs `--build`, not a restart — Next inlines
the origin into the client bundle at build time.

## Backups

This is the one thing a managed database was doing for you that now nobody
does. The corpus is synthetic and regenerable, but conversations, bookings
and audit rows are not.

```bash
docker compose exec -T db pg_dump -U carecopilot carecopilot | gzip > backup-$(date +%F).sql.gz
```

Worth a cron entry. The Neo4j volume needs no backup — it is a derived
projection, rebuilt by `scripts/build_kg.py` from PostgreSQL in seconds.

## Failure modes worth recognising

**The site times out, no logs anywhere.** One of the two firewalls. The cloud
Security List is the one people forget, the host iptables is the one they
don't know exists.

**Caddy logs a certificate failure.** `CARECOPILOT_DOMAIN` does not resolve
to this machine, or port 80 is not reachable — the HTTP-01 challenge needs
it even though nothing is served there. Fix DNS first; Let's Encrypt's rate
limit is five duplicate certificates per week, and restart loops reach it in
an afternoon.

**Startup fails with "Missing required configuration in production".**
`app/config.py` refusing to boot without a secret. That is the check working.
Supply the named variable; do not drop `ENVIRONMENT`.

**Answers work but citations are empty.** The RAG index is empty. Check the
`init` container's logs — `ingest_documents.py` runs there.

**Every knowledge-graph question says the graph holds nothing.** `init-kg`
failed or Neo4j is down. `docker compose logs init-kg`, then
`docker compose run --rm init-kg`.

**429 with `demo_quota_exhausted`.** The global daily budget, not a
per-visitor limit. Raise `RATE_LIMIT_DAILY_BUDGET` or accept it — the refusal
tells the visitor to run the project locally, which is true.
