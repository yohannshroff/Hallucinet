# Manual setup: Neo4j (needed starting Week 3)

Not installed by any script in this repo — run these yourself when you
reach Week 3 (KG ingestion).

## Option A: Docker (recommended)

```bash
docker run -d \
  --name hallucinet-neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/changeme \
  neo4j:5
```

- Browser UI: http://localhost:7474
- Bolt URI: `bolt://localhost:7687`
- Default credentials: `neo4j` / `changeme` — matches `.env.example`, change
  both if you use a different password.

## Option B: Neo4j Desktop

Download from https://neo4j.com/download/, create a new local DBMS, start
it, and note the bolt URI and password it gives you into your `.env`.

## After it's running

Copy `.env.example` to `.env` and fill in `NEO4J_URI`, `NEO4J_USER`,
`NEO4J_PASSWORD` to match whichever option you used. The Week 3 `neo4j`
Python driver (commented out in `requirements.txt` until then) will read
these via `python-dotenv`.
