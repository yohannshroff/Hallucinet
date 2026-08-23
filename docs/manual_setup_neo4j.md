# Manual setup: Neo4j (needed starting Part 3)

Not installed by any Part 1-2 script — set this up when you reach Part 3
(KG ingestion). This project's dev instance was set up with Option A below.

## Option A: Homebrew (macOS, what this repo's dev instance uses)

```bash
brew install neo4j

# Set credentials BEFORE the first start -- this only takes effect pre-first-boot.
neo4j-admin dbms set-initial-password 'your-password-here'

brew services start neo4j
```

- Browser UI: http://localhost:7474
- Bolt URI: `bolt://localhost:7687`
- Default user: `neo4j` (password is whatever you set above)
- Stop with `brew services stop neo4j`; check status with `brew services info neo4j`.

## Option B: Docker

```bash
docker run -d \
  --name hallucinet-neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/changeme \
  neo4j:5
```

Same ports/URIs as Option A.

## Option C: Neo4j Desktop

Download from https://neo4j.com/download/, create a new local DBMS, start
it, and note the bolt URI and password it gives you into your `.env`.

## After it's running

Copy `.env.example` to `.env` (if you haven't already) and fill in
`NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` to match whichever option you
used. `scripts/common.py` loads these via `python-dotenv` and every `kg/`
script reads the connection config from there.

Then:

```bash
python kg/load_graph.py
python kg/validate_graph.py --all
```
