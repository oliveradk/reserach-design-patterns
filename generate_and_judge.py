# Pattern for generate-then-judge pipelines using safety-tooling's InferenceAPI.
# Generates a response, then judges it with a binary classifier (yes/no).
# The batched version runs both steps within the same semaphore slot.

import asyncio

from safetytooling.apis import InferenceAPI
from safetytooling.data_models import ChatMessage, MessageRole, Prompt, LLMResponse
from tqdm.asyncio import tqdm_asyncio

from judges import binary_judge


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


async def pirate_judge(
    api: InferenceAPI,
    model: str,
    gen_messages: list[dict[str, str]],
    gen_response: str,
) -> dict:
    """Binary judge for pirate-likeness."""
    system_prompt = (
        "You are an LLM classifying whether responses are pirate-like."
    )
    user_template = (
        "Determine whether the following response to the given prompt is pirate-like "
        "(uses pirate speak, pirate mannerisms, pirate vocabulary, etc.).\n"
        "\n"
        "Respond with ONLY 'yes' or 'no'.\n"
        "\n"
        "Prompt: {prompt}\n"
        "Response: {response}"
    )

    first_user = next(m["content"] for m in gen_messages if m["role"] == "user")
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_template.format(prompt=first_user, response=gen_response)},
    ]
    return await binary_judge(api, model, messages, tag="pirate")


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
        return {**gen_result, "label": judge_result["label"], "judge": judge_result}

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
    with open("output/judge_prompts.txt", "w") as f:
        for i, r in enumerate(results):
            f.write(f"=== Result {i} ===\n")
            f.write(f"Judge system: {r['judge']['messages'][0]['content']}\n\n")
            f.write(f"Judge user:\n{r['judge']['messages'][1]['content']}\n\n")
            f.write(f"Classification: {r['judge']['label']}\n")
            f.write(f"\n")

    for r in results:
        print(f"  Q: {r['messages'][-1]['content']}")
        print(f"  A: {r['response']}")
        print(f"  Pirate? {r['judge']['label']} (tag: {r['judge']['tag']})")
        print()


if __name__ == "__main__":
    asyncio.run(main())
