import asyncio
import os
import json

from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

async def generate(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore, 
    model: str, 
    messages: list[dict[str, str]],
    generate_kwargs: dict
) -> dict:
    async with semaphore:
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            **generate_kwargs
        ) 
        return {"messages": messages, "response": response.choices[0].message.content}

async def generate_batch(
    client: AsyncOpenAI,
    model: str, 
    messages_batch: list[list[dict[str, str]]], 
    max_workers: int = 32,
    generate_kwargs: dict = {},
):
    semaphore = asyncio.Semaphore(max_workers)
    tasks = [
        asyncio.create_task(
            generate(client,semaphore, model, messages, generate_kwargs)
        ) 
        for messages in messages_batch
    ]
    results = await tqdm_asyncio.gather(*tasks, desc="Querying")

    return results

async def generate_batch_iterator(
    client: AsyncOpenAI,
    model: str, 
    messages_batch: list[list[dict[str, str]]], 
    max_workers: int = 32,
    generate_kwargs: dict = {},
    desc: str = "generating",
):
    semaphore = asyncio.Semaphore(max_workers)
    
    # Tag each task with its original index
    async def indexed_generate(i: int, messages: list[dict[str, str]]) -> tuple[int, dict]:
        result = await generate(client, semaphore, model, messages, generate_kwargs)
        return i, result

    tasks = [
        asyncio.create_task(indexed_generate(i, messages)) 
        for i, messages in enumerate(messages_batch)
    ]

    for finished in tqdm_asyncio.as_completed(tasks, total=len(tasks), desc=desc):
        i, result = await finished
        yield i, result
    

def jsonl_append(path: str, data: dict):
    with open(path, "a") as f:
        f.write(json.dumps(data) + "\n")


async def generate_batch_write_incremental(
    client: AsyncOpenAI,
    model: str, 
    messages_batch: list[list[dict[str, str]]], 
    max_workers: int = 32,
    generate_kwargs: dict = {},
    output_path: str = "results.jsonl",
):
    results = {}
    for i, result in generate_batch_iterator(client, model, messages_batch, max_workers, generate_kwargs):
        jsonl_append(output_path, {**result, "index": i})
        results[i] = result
    return [results[i] for i in sorted(results.keys())]

async def main():
    from dotenv import load_dotenv
    load_dotenv()

    results_1 = await generate_batch_write_incremental(
        client=AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY")),
        model="gpt-4.1-mini",
        messages_batch=[
            [{"role": "user", "content": "Hello, how are you?"}],
            [{"role": "user", "content": "What is the capital of France?"}],
        ],
        generate_kwargs={
            "temperature": 0.7,
            "max_tokens": 16
        },
        output_path="results_1.jsonl",
    )
    results_2 = await generate_batch(
        client=AsyncOpenAI(api_key=os.getenv("OPENROUTER_API_KEY")),
        model="deepseek/deepseek-chat-v3.1",
        messages_batch=[
            [{"role": "user", "content": "Hello, how are you?"}],
            [{"role": "user", "content": "What is the capital of France?"}],
        ],
        generate_kwargs={
            "temperature": 0.7,
            "max_tokens": 16
        },
        output_path="results_2.jsonl",
    )

if __name__ == "__main__":
    asyncio.run(main())