# Model API Query Patterns

Two common patterns for making many async API calls with progress tracking.

## Pattern 1: Incremental Writes with `asyncio.as_completed`

Use when you want to **stream results to disk as they finish** — good for long-running jobs where you want partial results if the process crashes.

```python
import asyncio
import json
from pathlib import Path

from openai import AsyncOpenAI
from tqdm.auto import tqdm


async def query_one(
    client: AsyncOpenAI,
    prompt: str,
    semaphore: asyncio.Semaphore,
) -> dict:
    async with semaphore:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        return {"prompt": prompt, "response": response.choices[0].message.content}


async def main(prompts: list[str], output_path: str = "results.jsonl"):
    client = AsyncOpenAI()
    semaphore = asyncio.Semaphore(32)
    output = Path(output_path)
    output.write_text("")  # clear file

    tasks = [asyncio.create_task(query_one(client, p, semaphore)) for p in prompts]

    results = []
    for finished in tqdm(
        asyncio.as_completed(tasks),
        total=len(tasks),
        desc="Querying",
    ):
        try:
            result = await finished
            results.append(result)
            # write immediately — survives crashes
            with output.open("a") as f:
                f.write(json.dumps(result) + "\n")
        except Exception as e:
            print(f"Failed: {e}")

    return results
```

**Key points:**
- `asyncio.create_task` launches all requests concurrently (bounded by the semaphore).
- `asyncio.as_completed` yields futures in **completion order**, not submission order.
- Each result is appended to the JSONL file the moment it's ready.
- `tqdm` wraps the iterator to show a progress bar.
- If the process dies halfway, you still have all completed results on disk.

**When to use:** Long jobs, expensive API calls, anything where losing progress would hurt.

## Pattern 2: Batch Gather with `tqdm_asyncio.gather`

Use when you just want **all results at the end** — simpler code, fine for shorter jobs.

```python
import asyncio

from openai import AsyncOpenAI
from tqdm.asyncio import tqdm as tqdm_asyncio


async def query_one(
    client: AsyncOpenAI,
    prompt: str,
    semaphore: asyncio.Semaphore,
) -> dict:
    async with semaphore:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        return {"prompt": prompt, "response": response.choices[0].message.content}


async def main(prompts: list[str]):
    client = AsyncOpenAI()
    semaphore = asyncio.Semaphore(32)

    tasks = [asyncio.create_task(query_one(client, p, semaphore)) for p in prompts]
    results = await tqdm_asyncio.gather(*tasks, desc="Querying")

    return results
```

**Key points:**
- `tqdm_asyncio.gather` is a drop-in replacement for `asyncio.gather` that shows a progress bar.
- All results are returned together as a list once every task finishes.
- Less boilerplate than Pattern 1 — no manual iteration or file writing.

**When to use:** Shorter jobs, or when you only need results in memory (e.g. to process and save once at the end).

## Comparison

| | Pattern 1 (Incremental) | Pattern 2 (Gather) |
|---|---|---|
| Results on disk during run | Yes | No |
| Crash recovery | Partial results saved | Nothing saved |
| Code complexity | More | Less |
| Result order | Completion order | Submission order |
| Progress bar | `tqdm` + `as_completed` | `tqdm_asyncio.gather` |
