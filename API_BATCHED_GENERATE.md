# Model API Query Patterns

Two common patterns for making many async API calls with progress tracking.

## Pattern 1: Batch Generate (with `tqdm_asyncio.gather`)

Use when you just want **all results at the end** — simpler code, fine for shorter jobs.

```python
import asyncio

from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio


async def generate(
    client: AsyncOpenAI,
    prompt: str,
    semaphore: asyncio.Semaphore,
) -> dict:
    async with semaphore:
        response = await client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        return {"prompt": prompt, "response": response.choices[0].message.content}


async def generate_batch(prompts: list[str]):
    client = AsyncOpenAI()
    semaphore = asyncio.Semaphore(32)

    tasks = [asyncio.create_task(generate(client, p, semaphore)) for p in prompts]
    results = await tqdm_asyncio.gather(*tasks, desc="Querying")

    return results
```


## Pattern 2: Incremental Write Batch Generate (with `asyncio.as_completed` and `tqdm`)

Use when you want to **stream results to disk as they finish** — good for long-running jobs where you want partial results if the process crashes.

```python
import asyncio
import json
from pathlib import Path

from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio


async def generate(
    client: AsyncOpenAI,
    prompt: str,
    semaphore: asyncio.Semaphore,
) -> dict:
    async with semaphore:
        response = await client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        return {"prompt": prompt, "response": response.choices[0].message.content}


def append_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False))
        f.write("\n")


async def generate_batch(prompts: list[str], output_path: str = "results.jsonl"):
    client = AsyncOpenAI()
    semaphore = asyncio.Semaphore(32)
    output = Path(output_path)
    output.write_text("")  # clear file

    tasks = [asyncio.create_task(generate(client, p, semaphore)) for p in prompts]

    results = []
    for finished in tqdm_asyncio.as_completed(
        tasks, total=len(tasks), desc="generating"
    ):
        try:
            result = await finished
            results.append(result)
            append_jsonl(output, result)
        except Exception as e:
            print(f"Failed: {e}")

    return results
```