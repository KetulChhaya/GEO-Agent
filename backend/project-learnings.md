# CiteSight Learning Summary

## ASYNC DB FUNDAMENTALS

### Sync vs Async

**Sync (Blocking):**
```python
import time

def fetch_from_db():
    print("Waiting for DB...")
    time.sleep(2)  # DB takes 2 seconds - CODE WAITS HERE
    print("Got data!")
    return "data"

# Processing 3 requests:
# Request 1: 0s-2s (waits)
# Request 2: 2s-4s (waits for request 1)
# Request 3: 4s-6s (waits for request 2)
# Total: 6 seconds
```

**Async (Non-Blocking):**
```python
import asyncio

async def fetch_from_db():
    print("Waiting for DB...")
    await asyncio.sleep(2)  # "I'm waiting, you do other stuff"
    print("Got data!")
    return "data"

# Processing 3 requests concurrently:
# Request 1: 0s start, 2s end
# Request 2: 0.01s start, 2s end (started later but overlapped)
# Request 3: 0.02s start, 2s end
# Total: ~2 seconds (all happen together)
```

**Key insight:** Async doesn't make DB faster. It lets server handle OTHER requests while waiting.

### AsyncGenerator

Function that yields values asynchronously:
```python
async def async_gen():
    yield value1
    yield value2

# Use it:
async for x in async_gen():
    print(x)
```

---

## DATABASE & ORM CONCEPTS

### Session Management (`backend/app/db/session.py`)

```python
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

engine = create_async_engine(settings.database_url)
async_session = async_sessionmaker(engine, expire_on_commit=False)

async def get_session() -> AsyncGenerator[AsyncSession]:
    async with async_session() as session:
        yield session  # Used by FastAPI dependency injection
```

**Key Points:**
- `create_async_engine` = connection pool (non-blocking)
- `async_sessionmaker` = factory for creating sessions
- `get_session()` = yields one session per request, auto-cleanup
- `await` needed on all DB calls

### Alembic Migrations (`backend/app/db/migrations/versions/0001_initial.py`)

Alembic is a Version control for database schema (like Git for tables/columns).

**env.py file:**
- Reads database settings
- Compares current DB vs target (Base.metadata from models)
- Creates/alters tables
- Runs inside transaction (rollback on error)

**Workflow:**
1. Add column to model: `age: int`
2. Run: `alembic revision --autogenerate -m "add age"`
3. Run: `alembic upgrade head` → applies migration

### __pycache__ Directories

**What:** Compiled Python bytecode cache.

**Why:** Speed. Python compiles `.py` → bytecode, caches in `__pycache__` for faster second run.

**When updates:** When you modify `.py` file (Python checks timestamp).

**Safe to delete?** YES. Python regenerates automatically. Add to `.gitignore`.

---

## TEXT & EMBEDDINGS

### Embeddings Basics

embeddings: Convert text → vector (list of numbers).

```python
"The cat sat on mat"     → [0.1, -0.23, 0.45, ..., 0.12]  (1536 dims)
"A feline was on rug"    → [0.11, 0.19, -0.29, ..., 0.11]  (similar vectors)
```

Vector distance = text similarity. Similar meaning = similar vectors.

### 1536-d Vector

**1536-d = 1536 dimensions = 1536 numbers**

```python
vector = [0.1, -0.23, 0.45, ..., 0.12]
         └──────────── 1536 values ────────────┘
```

**Key constraint:** Model always outputs 1536 dims, regardless of input:
- 1 word → 1536 dims
- 100 words → 1536 dims
- 10k words → 1536 dims (some info lost due to compression)
- 100k words → ERROR (exceeds API limit ~8k tokens)

**Cannot change easily:** Would need different model + DB schema migration + re-embed everything.

### Text Chunking (`backend/app/graph/nodes/chunk_and_embed.py`)

**Problem:** Long texts exceed embedding API limits. Need to split smartly.

**Solution:**
```python
CHUNK_SIZE = 800      # 800-char chunks
CHUNK_OVERLAP = 100   # Next chunk starts 100 chars before previous ends
```

**Why overlap?** Sentences don't split in half. Each chunk has context before AND after.

**Example:**
```
Text: "...We serve 10,000+ customers globally. Founded in 2010. Our values..."

Chunk 1 (0-800):   "...customers globally. Founded in 2010. Our values..."
Chunk 2 (700-1500): "Founded in 2010. Our values are transparency..."
                     └─ overlapped, sentence preserved ─┘
```

**Benefit:** Embeddings capture full context. Better similarity search.

### OpenAI Embeddings Client (`backend/app/llm/embeddings.py`)

**Purpose:** Wrapper around OpenAI embedding API.

**Key features:**
- Batch 100 texts per request (MAX_BATCH = 100)
- Retry 3 times with exponential backoff (2s, 4s, 8s)
- Track tokens against budget cap

**Batching example (MAX_BATCH = 100):**
```python
# You have 250 texts to embed
texts = [text1, text2, ..., text250]

vectors = await client.embed_all(texts)

# Behind the scenes:
# Batch 1: texts[0:100] → 1 API call → 45 tokens
# Batch 2: texts[100:200] → 1 API call → 45 tokens
# Batch 3: texts[200:250] → 1 API call → 22 tokens
# Total: 3 API calls, 112 tokens

# If MAX_BATCH was 50:
# Batch 1: texts[0:50] → 25 tokens
# Batch 2: texts[50:100] → 25 tokens
# Batch 3: texts[100:150] → 25 tokens
# Batch 4: texts[150:200] → 25 tokens
# Batch 5: texts[200:250] → 22 tokens
# Total: 5 API calls, 122 tokens (more calls, slightly more tokens but safer)
```

**EMBED_DIMENSIONS (1536):**
```python
DEFAULT_EMBED_MODEL = "text-embedding-3-small"  # Always outputs 1536 dims
EMBED_DIMENSIONS = 1536

# Every embedding is exactly 1536 numbers:
vector = [0.1, -0.23, 0.45, ..., 0.12]  # ← 1536 floats

# Database stores as VECTOR(1536):
CREATE TABLE chunks (
    embedding VECTOR(1536)  -- Stores 1536-d vectors
);

# Cannot change easily:
# - Different model (text-embedding-3-large) → 3072 dims
# - Requires: DB schema migration + re-embedding everything
# - Costs: 3x more tokens (re-embed all chunks)
```

**Retry decorator:**
```python
@retry(
    stop=stop_after_attempt(3),                          # Max 3 tries
    wait=wait_exponential(multiplier=1, min=2, max=20),  # Wait: 2s, 4s, 8s...
    reraise=True,                                         # If all fail, raise error
)
```

Why? Network fails sometimes. Wait longer each time so server can recover.

**Token tracking with budget enforcement:**
```python
tracker = TokenBudgetTracker(cap=500)  # Budget: 500 tokens

# Batch 1:
texts_batch1 = ["text1", "text2", ..., "text100"]  (100 texts)
vectors, tokens = await client._embed_batch(texts_batch1)
# tokens = 50 (OpenAI charged 50 tokens)
await tracker.record(50)
# tracker.used = 50, tracker.remaining = 450 ✓

# Batch 2:
texts_batch2 = ["text101", ..., "text200"]  (100 texts)
vectors, tokens = await client._embed_batch(texts_batch2)
# tokens = 50
await tracker.record(50)
# tracker.used = 100, tracker.remaining = 400 ✓

# ... continue batches ...

# Batch 9:
texts_batch9 = [...]  (100 texts)
vectors, tokens = await client._embed_batch(texts_batch9)
# tokens = 450 (large batch)
await tracker.record(450)
# tracker.used = 550, tracker.remaining = -50 ✗
# Raises BudgetExceeded(used=550, requested=450, cap=500)
# Embedding stops immediately
```

**In embed_all() method:**
```python
async def embed_all(self, texts: list[str]) -> list[list[float]]:
    vectors: list[list[float]] = []
    for start in range(0, len(texts), MAX_BATCH):
        batch = texts[start : start + MAX_BATCH]
        
        # Check budget BEFORE fetching
        if self._tracker.remaining <= 0:
            raise BudgetExceeded(...)  # Stop here, don't waste request
        
        batch_vectors, tokens = await self._embed_batch(batch)
        await self._tracker.record(tokens)  # Record AFTER success
        vectors.extend(batch_vectors)
    
    return vectors  # Return only vectors (tokens already tracked internally)
```

**Return types differ:**
- `_embed_batch()` returns: `(vectors, tokens_used)` — low-level, raw result
- `embed_all()` returns: `vectors` only — high-level, tracks internally

---

## CITESIGHT PHASE 1 ARCHITECTURE

### Graph Flow

```
START
  ↓
crawl_site (discover & fetch pages)
  ↙ (failed)           ↘ (continue)
 END              chunk_and_embed (split, embed, store)
                   ↓
                  END
```

### crawl_site Node (`backend/app/graph/nodes/crawl_site.py`)

**Purpose:** Crawl website, extract content, detect brand name.

**Full workflow example:**

```python
# Input:
state.url = "https://example.com"
state.errors = []

# Step 1: Load robots.txt
robots = await _load_robots(client, "https://example.com", errors)
# robots.txt parsed. Can now check if URLs are crawlable.

# Step 2: Discover URLs (try sitemap first)
candidates = await _discover_urls(
    client, "https://example.com", robots, max_pages=50
)
# Tried sitemap.xml → found 10 URLs
# All same-domain, robots-allowed, no duplicates
# Capped at 50 max
# candidates = ["https://example.com", "/about", "/products", "/blog/post1", ...]

# Step 3: Fetch pages (concurrent, max 5 at a time)
sem = asyncio.Semaphore(5)  # Limit to 5 concurrent requests
results = await asyncio.gather(
    *(_fetch_page(client, url, sem, errors) for url in candidates)
)
# All 10 URLs fetched (in batches of 5)
# results = [(url, html), (url, html), None, (url, html), ...]
# None for failed/non-HTML pages

# Step 4: Extract content & metadata
pages = []
brand_name = None
for url, html in results:
    if html is None:
        continue  # Skip failed pages
    
    title, content = _extract_main(html)
    # title = page <title>
    # content = main text (no nav/footer/scripts)
    
    if not content:  # E.g., JS-only shell with no text
        continue  # Skip
    
    pages.append(PageMeta(
        url=url,
        title=title,
        content=content,
        word_count=len(content.split())
    ))
    
    # Extract brand from homepage
    if brand_name is None and url == "https://example.com":
        brand = _extract_brand(html)  # og:site_name > og:title > <title>
        # brand = "Acme Corp"
        brand_name = brand

# Step 5: Quality gate
usable = sum(1 for p in pages if p.word_count >= 200)
# e.g., 7 out of 10 pages have 200+ words
if usable < 3:  # MIN_USABLE_PAGES = 3
    errors.append(f"insufficient content: found {usable} usable page(s), need 3")
    status = "failed_insufficient_content"
else:
    status = None  # Normal, continue to chunk_and_embed

# Result:
return {
    "pages": [
        PageMeta(url="https://example.com", title="Acme", content="...", word_count=500),
        PageMeta(url="https://example.com/about", title="About", content="...", word_count=350),
        ...
    ],
    "brand_name": "Acme Corp",
    "errors": [],
    "status": None  # or "failed_insufficient_content"
}
```

**Key functions:**

1. **Discovery:** `_discover_urls()` → list of URLs
   - Try sitemap first (fastest)
   - Fallback to link crawl if <3 URLs found
   - Respect robots.txt
   - Deduplicate, cap at max_pages

2. **Fetching:** `_fetch_page()` → (url, html) or None
   - Semaphore limits concurrency (MAX 5 concurrent)
   - Recoverable errors logged, not fatal

3. **Content extraction:**
   - `_extract_main()` → (title, main_text) — remove boilerplate (nav, footer, scripts)
   - `_extract_brand()` → brand_name — og:site_name > og:title > <title>

4. **Quality gate:** Check if enough usable content
   ```python
   usable = count(pages where word_count >= MIN_USABLE_WORDS)
   if usable < MIN_USABLE_PAGES:  # e.g., 3 pages with 200+ words
       status = "failed_insufficient_content"
   ```

**Result:** List of PageMeta objects + brand name + errors.

### chunk_and_embed Node (`backend/app/graph/nodes/chunk_and_embed.py`)

**Purpose:** Split pages into chunks, embed, store in DB.

**Key steps:**

1. Split text into chunks:
   ```python
   splitter = RecursiveCharacterTextSplitter(
       chunk_size=800, chunk_overlap=100
   )
   
   page_content = "Company was founded in 2010. We focus on AI research. Our mission is to make AI accessible. We have offices in California, New York, and London. Our team has 50 engineers. We raised $100M in funding. Our products include language models and embeddings. We serve 10,000+ customers globally..."
   
   chunks = splitter.split_text(page_content)
   # Result:
   # chunks[0] = "Company was founded in 2010. We focus on AI research... [800 chars total]"
   # chunks[1] = "...research and innovation. We have 50 engineers... [800 chars, 100-char overlap with chunks[0]]"
   # chunks[2] = "...engineers. We raised $100M in funding... [remaining text]"
   ```
   
   **Why overlap?** Sentence "Our products include language models and embeddings" isn't split across chunk boundary.

2. Track metadata for each chunk:
   ```python
   texts = [
       "Company was founded in 2010. We focus on AI research...",  # chunk 0
       "...research and innovation. We have 50 engineers...",      # chunk 1
       "...engineers. We raised $100M in funding..."               # chunk 2
   ]
   
   metadatas = [
       {"url": "https://example.com/about", "title": "About Us", "chunk_index": 0},
       {"url": "https://example.com/about", "title": "About Us", "chunk_index": 1},
       {"url": "https://example.com/about", "title": "About Us", "chunk_index": 2},
   ]
   ```

3. Embed all chunks:
   ```python
   vectors = await embeddings_client.embed_all(texts)
   # Result: 3 vectors, each 1536-d
   # vectors[0] = [0.1, -0.23, 0.45, ..., 0.12]  (1536 floats)
   # vectors[1] = [0.11, 0.19, -0.29, ..., 0.11]
   # vectors[2] = [0.12, -0.21, 0.48, ..., 0.14]
   
   tokens_used = 45  # OpenAI charged 45 tokens for this batch
   ```

4. Store in DB:
   ```python
   # Upsert audit row (idempotent - won't error if exists)
   INSERT INTO audits (run_id, url) 
   VALUES ("audit_12345", "https://example.com")
   ON CONFLICT DO NOTHING
   
   # Insert chunks with embeddings and metadata
   INSERT INTO chunks (run_id, namespace, content, embedding, metadata_)
   VALUES 
       ("audit_12345", "site:audit_12345", 
        "Company was founded in 2010. We focus on AI research...", 
        ARRAY[0.1, -0.23, 0.45, ...],  -- 1536-d vector
        '{"url": "https://example.com/about", "title": "About Us", "chunk_index": 0}'),
       
       ("audit_12345", "site:audit_12345",
        "...research and innovation. We have 50 engineers...",
        ARRAY[0.11, 0.19, -0.29, ...],  -- 1536-d vector
        '{"url": "https://example.com/about", "title": "About Us", "chunk_index": 1}'),
       
       -- ... more chunks
   ```

**Result:** Chunks stored in `pgvector` column for similarity search.

**Example: Later similarity search**
```sql
SELECT content, metadata_ 
FROM chunks 
WHERE namespace = "site:audit_12345"
ORDER BY embedding <-> '[0.12, -0.18, 0.3, ...]'::vector  -- distance to query
LIMIT 5;
-- Returns 5 most similar chunks to query vector
```

---

## 5. LANGGRAPH FUNDAMENTALS

### Graph Builder (`backend/app/graph/builder.py`)

**Purpose:** Assemble nodes into executable workflow.

**Key components:**

1. **Conditional router:** `route_after_crawl(state)`
   ```python
   # If crawl found insufficient content, fail
   return "end" if state.status.startswith("failed") else "continue"
   ```

2. **Build graph:**
   ```python
   builder = StateGraph(AuditState)
   builder.add_node("crawl_site", crawl_site)
   builder.add_node("chunk_and_embed", chunk_and_embed)
   builder.add_edge(START, "crawl_site")
   builder.add_conditional_edges(
       "crawl_site",
       route_after_crawl,
       {"continue": "chunk_and_embed", "end": END}
   )
   return builder.compile(checkpointer=checkpointer)
   ```

3. **Type signature:**
   ```python
   def build_graph(
       checkpointer: BaseCheckpointSaver[Any] | None = None,
   ) -> CompiledStateGraph[AuditState, Any, AuditState, AuditState]:
   ```

   - Input: `AuditState`
   - Output: `AuditState` (updated)
   - Checkpointer: optional (enables resume-on-crash)

### Checkpointing

**Problem:** Long audits crash mid-run. Lose all progress.

**Solution:** Save state after each node to database.

**`_postgres_dsn()`:** Convert URL format
```python
# SQLAlchemy format:  "postgresql+asyncpg://..."
# LangGraph format:   "postgresql://..."
# Function strips "+asyncpg" suffix
```

**`open_checkpointer()`:** Context manager for safe DB access
```python
@asynccontextmanager
async def open_checkpointer() -> AsyncIterator[AsyncPostgresSaver]:
    async with AsyncPostgresSaver.from_conn_string(dsn) as saver:
        await saver.setup()  # Create checkpoint tables
        yield saver  # Give to user
        # Auto-cleanup on exit
```

**Usage & Resume Example:**

**Run 1 (crashes after crawl_site):**
```python
state = AuditState(url="https://example.com", run_id="audit_123")

async with open_checkpointer() as saver:
    graph = build_graph(checkpointer=saver)
    
    # Checkpoint tables created automatically:
    # - checkpoints table (saves state)
    # - checkpoint_writes table (tracks writes)
    # - checkpoint_blobs table (stores large data)
    
    result = await graph.ainvoke(
        state, 
        {"configurable": {"thread_id": "audit_123"}}
    )
    # Progress:
    # 1. crawl_site starts
    # 2. crawl_site completes → state saved to DB
    # 3. router checks state.status
    # 4. chunk_and_embed starts
    # 5. [CRASH] - server dies mid-embed
```

**Run 2 (resume from crash):**
```python
# Same run_id (thread_id) tells LangGraph to resume
result = await graph.ainvoke(
    state,
    {"configurable": {"thread_id": "audit_123"}}  # Same ID
)
# LangGraph reads checkpoint from DB
# Sees: "crawl_site completed, last node was crawl_site"
# Skips crawl_site (already done)
# Resumes from chunk_and_embed
# MUCH faster!
```

**Without checkpointer:**
```python
graph = build_graph()  # checkpointer=None
# Crash? Run again with same state
# Starts over: re-crawls entire site (slow, wasteful)
```

**Checkpoint workflow visualization:**
```
RUN 1:
  crawl_site → [state saved to DB]
  router → continue
  chunk_and_embed → [CRASH]

RUN 2 (same thread_id):
  [Read last checkpoint from DB]
  "crawl_site already done, last node was crawl_site"
  → Skip crawl_site
  → Resume chunk_and_embed
  → Complete
```

---

## 6. KEY DEPENDENCIES

### LangChain/LangGraph Stack

- **langchain-text-splitters:** Split long text into chunks smartly
- **langgraph:** Workflow engine for multi-step LLM tasks
- **langgraph-checkpoint-postgres:** Save/restore state to DB for resumability

### Database Stack

- **FastAPI:** Web framework (async-first)
- **SQLAlchemy:** ORM + async support (`async_sessionmaker`, `AsyncSession`)
- **Alembic:** Database migration tool
- **asyncpg/psycopg:** Database drivers (async/sync)
- **pgvector:** PostgreSQL extension for vector storage (embeddings)

### LLM Stack

- **OpenAI:** Embeddings API (text-embedding-3-small, 1536-d vectors)
- **Tenacity:** Retry library with exponential backoff

---

## 7. PHASE 1 DATA FLOW

```
1. INPUT: Company URL
   └─→ state.url = "https://example.com"

2. CRAWL_SITE NODE:
   └─→ Discover URLs (sitemap or crawl)
   └─→ Fetch pages (concurrent, semaphore-limited)
   └─→ Extract content (strip boilerplate)
   └─→ Check quality (need 3+ pages with 200+ words)
   └─→ OUTPUT: pages[], brand_name, errors[], status

3. ROUTER (route_after_crawl):
   └─→ If status="failed_*" → go to END
   └─→ Else → continue to chunk_and_embed

4. CHUNK_AND_EMBED NODE:
   └─→ Split pages into 800-char chunks (100-char overlap)
   └─→ Track metadata (url, title, chunk_index)
   └─→ Embed all chunks → 1536-d vectors
   └─→ Store in DB (chunks table with pgvector)
   └─→ OUTPUT: chunk_count

5. END
```

---

## 8. QUICK REFERENCE

### Common Terms

| Term | Meaning |
|------|---------|
| **Async** | Non-blocking I/O. Yield CPU while waiting. |
| **Semaphore** | Limit concurrent operations (e.g., max 5 requests). |
| **Checkpointer** | Save/restore state for crash recovery. |
| **Vector** | List of numbers representing text meaning. |
| **Embedding** | Process of converting text → vector. |
| **Chunk** | Split of long text (800 chars here). |
| **Boilerplate** | Navigation, footer, scripts (not main content). |
| **Robots.txt** | File that tells crawlers what pages they can access. |
| **Sitemap** | XML file listing all pages on site. |
| **Idempotent** | Safe to run multiple times (no side effects). |

### File Map

- `session.py` → Async DB session management
- `crawl_site.py` → Web crawling + content extraction
- `chunk_and_embed.py` → Text splitting + vectorization
- `embeddings.py` → OpenAI embedding client wrapper
- `builder.py` → LangGraph workflow assembly
- `env.py` (alembic) → Migration configuration

---

## 9. LEARNING OUTCOMES

After this session, you understand:

✓ Async/await and why it matters for I/O-bound tasks
✓ How SQLAlchemy async sessions work
✓ How Alembic tracks schema changes
✓ How text embeddings work (1536-d vectors)
✓ Text chunking with overlap
✓ Web crawling with politeness (robots.txt, rate limits)
✓ LangGraph nodes and conditional routing
✓ Checkpoint-based resume-on-crash
✓ Token tracking and budget enforcement
