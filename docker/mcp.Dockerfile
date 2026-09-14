# CareCopilot MCP server.
#
# Deliberately thin. This process holds no database credentials and no
# service token — it proxies the backend API using the *caller's* JWT, which
# is what makes "the MCP surface cannot bypass authorization" a structural
# property rather than a policy. The image reflects that: no database
# driver, no models, no migrations, nothing but an HTTP client and the tool
# definitions.

FROM python:3.14-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN pip install --no-cache-dir "mcp>=2.2,<3" "httpx>=0.27" \
    && useradd --create-home --uid 10002 mcp

WORKDIR /app
COPY --chown=mcp:mcp mcp-server/ /app/
USER mcp

EXPOSE 8100

CMD ["python", "server.py"]
