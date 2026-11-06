# Optional (Task 12.5): the edge agent as a container, only for a Linux laptop.
# On Windows install it as a native service instead (README.md); Docker Desktop is too heavy for an old Celeron.
# Build from the repo root:  docker build -f edge_agent/Dockerfile -t cuy-edge-agent .
# Run:  docker run -d --restart unless-stopped --env-file edge_agent/.env --name cuy-edge-agent cuy-edge-agent
FROM python:3.14-slim
WORKDIR /app
RUN groupadd --system edge && useradd --system --gid edge --home-dir /app edge
COPY edge_agent/requirements.txt ./edge_agent/requirements.txt
RUN pip install --no-cache-dir -r edge_agent/requirements.txt
COPY edge_agent ./edge_agent
# the settings come from --env-file; .dockerignore keeps edge_agent/.env (the API key) out of the image
USER edge
ENV PYTHONUNBUFFERED=1
CMD ["python", "-m", "edge_agent"]
