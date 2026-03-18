# Patterns for generating with safety-tooling's InferenceAPI.
# Wraps InferenceAPI to match the generate / generate_batch interface
# from api_batched_generate.py.
#
# Key difference: InferenceAPI has built-in caching, concurrency control,
# and provider routing, so we don't need manual semaphores or AsyncOpenAI.

import asyncio

from safetytooling.apis import InferenceAPI
from safetytooling.data_models import ChatMessage, MessageRole, Prompt, LLMResponse
from tqdm.asyncio import tqdm_asyncio


async def generate(
    api: InferenceAPI,
    model: str,
    messages: list[dict[str, str]],
    **kwargs,
) -> dict:
    """Single generation. Returns dict with messages and response text."""
    prompt = Prompt(messages=[
        ChatMessage(role=MessageRole(m["role"]), content=m["content"])
        for m in messages
    ])
    responses: list[LLMResponse] = await api(model_id=model, prompt=prompt, **kwargs)
    return {
        "messages": messages,
        "response": responses[0].completion,
    }


async def generate_batch(
    api: InferenceAPI,
    model: str,
    messages_batch: list[list[dict[str, str]]],
    generate_kwargs: dict | None = None,
    desc: str = "generating",
) -> list[dict]:
    """Batch generation with progress bar. Returns list of result dicts."""
    if generate_kwargs is None:
        generate_kwargs = {}

    # semaphore used internally
    tasks = [
        generate(api, model, messages, **generate_kwargs)
        for messages in messages_batch
    ]

    results = await tqdm_asyncio.gather(*tasks, desc=desc)
    return list(results)


async def main():
    from pathlib import Path
    from dotenv import load_dotenv

    load_dotenv()

    api = InferenceAPI(
        anthropic_num_threads=10,
        openai_num_threads=10,
        cache_dir=Path("output/cache"),
    )

    pirate_system = {"role": "system", "content": "You are a pirate. Respond in pirate speak."}

    # --- Single generate with system prompt ---
    result = await generate(
        api=api,
        model="gpt-4.1-mini",
        messages=[pirate_system, {"role": "user", "content": "What is the capital of France?"}],
        max_tokens=64,
    )
    print("Single generate (pirate):")
    print(f"  Response: {result['response']}")

    # --- Batch generate with system prompt ---
    messages_batch = [
        [pirate_system, {"role": "user", "content": "What is the capital of France?"}],
        [pirate_system, {"role": "user", "content": "What is the capital of Germany?"}],
        [pirate_system, {"role": "user", "content": "What is the capital of Japan?"}],
    ]

    print("\nBatch generate (pirate, first run):")
    results = await generate_batch(
        api=api,
        model="gpt-4.1-mini",
        messages_batch=messages_batch,
        generate_kwargs={"max_tokens": 64},
        desc="first run",
    )
    for r in results:
        print(f"  {r['messages'][1]['content']} -> {r['response']}")

    # --- Demo caching: second run should be instant ---
    print("\nBatch generate (pirate, cached run):")
    results = await generate_batch(
        api=api,
        model="gpt-4.1-mini",
        messages_batch=messages_batch,
        generate_kwargs={"max_tokens": 64},
        desc="cached run",
    )
    for r in results:
        print(f"  {r['messages'][1]['content']} -> {r['response']}")


if __name__ == "__main__":
    asyncio.run(main())
