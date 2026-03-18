# Pattern for generate-then-judge pipelines using safety-tooling's InferenceAPI.
# Generates a response, then judges it with a binary classifier (yes/no).
# The batched version runs both steps within the same semaphore slot.

import asyncio
import re

from safetytooling.apis import InferenceAPI
from safetytooling.data_models import ChatMessage, MessageRole, Prompt, LLMResponse
from tqdm.asyncio import tqdm_asyncio


def messages_to_prompt(messages: list[dict[str, str]]) -> Prompt:
    return Prompt(messages=[
        ChatMessage(role=MessageRole(m["role"]), content=m["content"])
        for m in messages
    ])


async def generate(
    api: InferenceAPI,
    model: str,
    messages: list[dict[str, str]],
    **kwargs,
) -> dict:
    """Single generation. Returns dict with messages and response text."""
    prompt = messages_to_prompt(messages)
    responses: list[LLMResponse] = await api(model_id=model, prompt=prompt, **kwargs)
    return {
        "messages": messages,
        "response": responses[0].completion,
    }


YES_NO_PATTERN = re.compile(r"\b(yes|no)\b", re.IGNORECASE)


async def binary_judge(
    api: InferenceAPI,
    model: str,
    messages: list[dict[str, str]],
    judge_tag: str = "",
) -> dict:
    """Generic binary judge. Returns classification (bool or None on parse error)."""
    prompt = messages_to_prompt(messages)
    responses: list[LLMResponse] = await api(model_id=model, prompt=prompt, temperature=0, max_tokens=1)
    raw = responses[0].completion
    match = YES_NO_PATTERN.search(raw)
    classification = match.group(1).lower() == "yes" if match else None
    return {
        "messages": messages,
        "response": raw,
        "classification": classification,
        "judge_tag": judge_tag,
    }




async def pirate_judge(
    api: InferenceAPI,
    model: str,
    gen_messages: list[dict[str, str]],
    gen_response: str,
) -> dict:
    """Binary judge for pirate-likeness. Uses jinja templates."""
    from pathlib import Path
    from jinja2 import Environment, FileSystemLoader

    templates_dir = Path(__file__).parent / "templates" / "pirate_judge"
    env = Environment(loader=FileSystemLoader(templates_dir))
    system_template = env.get_template("system.jinja")
    user_template = env.get_template("user.jinja")

    first_user = next(m["content"] for m in gen_messages if m["role"] == "user")
    messages = [
        {"role": "system", "content": system_template.render()},
        {"role": "user", "content": user_template.render(prompt=first_user, response=gen_response)},
    ]
    return await binary_judge(api, model, messages, judge_tag="pirate")


async def batched_generate_and_pirate_judge(
    api: InferenceAPI,
    generate_model: str,
    judge_model: str,
    messages_batch: list[list[dict[str, str]]],
    max_workers: int = 32,
    generate_kwargs: dict | None = None,
    desc: str = "generate & pirate judge",
) -> list[dict]:
    """Batch generate-then-pirate-judge. Both steps share a semaphore slot."""
    if generate_kwargs is None:
        generate_kwargs = {}

    semaphore = asyncio.Semaphore(max_workers)

    async def _throttled_generate_and_judge(messages):
        async with semaphore:
            gen_result = await generate(api, generate_model, messages, **generate_kwargs)
            judge_result = await pirate_judge(api, judge_model, messages, gen_result["response"])
        return {**gen_result, "judge": judge_result}

    results = await tqdm_asyncio.gather(
        *(_throttled_generate_and_judge(messages) for messages in messages_batch),
        desc=desc,
    )
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
    normal_system = {"role": "system", "content": "You are a helpful assistant."}

    messages_batch = [
        [pirate_system, {"role": "user", "content": "What is the capital of France?"}],
        [pirate_system, {"role": "user", "content": "What is the capital of Germany?"}],
        [normal_system, {"role": "user", "content": "What is the capital of Japan?"}],
    ]

    results = await batched_generate_and_pirate_judge(
        api=api,
        generate_model="gpt-4.1-mini",
        judge_model="gpt-4.1-mini",
        messages_batch=messages_batch,
        generate_kwargs={"max_tokens": 64},
    )
    for r in results:
        print(f"  Q: {r['messages'][-1]['content']}")
        print(f"  A: {r['response']}")
        print(f"  Pirate? {r['judge']['classification']} (tag: {r['judge']['judge_tag']})")
        print()


if __name__ == "__main__":
    asyncio.run(main())
