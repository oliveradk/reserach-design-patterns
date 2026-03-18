# Judge types using safety-tooling's InferenceAPI:
#   1. BinaryJudge            — yes/no classification via regex parsing
#   2. BinaryLogitJudge       — yes/no classification via token logprobs
#   3. ScoringLogitJudge      — numeric score (1-5) via token logprobs
#   4. MultiCriterionScoringJudge — multiple named scores parsed from XML tags

import asyncio
import math
import re

from safetytooling.apis import InferenceAPI
from safetytooling.apis.utils import binary_response_logit
from safetytooling.data_models import ChatMessage, MessageRole, Prompt, LLMResponse
from safetytooling.utils.math_utils import logsumexp


def messages_to_prompt(messages: list[dict[str, str]]) -> Prompt:
    return Prompt(messages=[
        ChatMessage(role=MessageRole(m["role"]), content=m["content"])
        for m in messages
    ])

def expected_score_from_logprobs(
    response: LLMResponse,
    min_score: int = 1,
    max_score: int = 5,
    token_idx: int = 0,
) -> float | None:
    """
    Returns the expected score from logprobs on integer tokens.
    Normalises probabilities over score tokens found in the logprobs.
    Returns None if no score tokens are found.
    """
    assert response.logprobs is not None

    score_tokens = {str(s) for s in range(min_score, max_score + 1)}

    final_logprobs = response.logprobs[token_idx]
    final_tokens = final_logprobs.keys()
    if set(t.strip() for t in final_tokens) & score_tokens == set():
        return None

    score_logprobs = {
        int(t.strip()): final_logprobs[t]
        for t in final_tokens if t.strip() in score_tokens
    }

    # Normalise in log-space using logsumexp, then compute expected score
    log_total = logsumexp(list(score_logprobs.values()))
    normalised = {s: math.exp(lp - log_total) for s, lp in score_logprobs.items()}

    return sum(s * p for s, p in normalised.items())


# ---------------------------------------------------------------------------
# 1. Binary judge — yes/no classification via regex parsing
# ---------------------------------------------------------------------------

YES_NO_PATTERN = re.compile(r"\b(yes|no)\b", re.IGNORECASE)


async def binary_judge(
    api: InferenceAPI,
    model: str,
    messages: list[dict[str, str]],
    tag: str = "",
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
        "tag": tag,
    }


# ---------------------------------------------------------------------------
# 2. Binary logit judge — uses logprobs on "yes"/"no" tokens
# ---------------------------------------------------------------------------

async def binary_logit_judge(
    api: InferenceAPI,
    model: str,
    messages: list[dict[str, str]],
    tag: str = "",
) -> dict:
    """Binary judge using logprobs. Returns logit (yes > no) and classification."""
    prompt = messages_to_prompt(messages)
    responses: list[LLMResponse] = await api(
        model_id=model, prompt=prompt, temperature=0, max_tokens=4, logprobs=5,
    )
    response = responses[0]

    logit = binary_response_logit(
        response=response,
        tokens1=("yes", "Yes", "YES", "y", "Y"),
        tokens2=("no", "No", "NO", "n", "N"),
        token_idx=0,
    )
    classification = logit > 0 if logit is not None else None

    return {
        "messages": messages,
        "response": response.completion,
        "classification": classification,
        "logit": logit,
        "tag": tag,
    }


# ---------------------------------------------------------------------------
# 2. Scoring logit judge — uses logprobs on score tokens (1-5)
# ---------------------------------------------------------------------------
async def scoring_logit_judge(
    api: InferenceAPI,
    model: str,
    messages: list[dict[str, str]],
    tag: str = "",
) -> dict:
    """Scoring judge using logprobs. Returns expected score from token probabilities."""
    prompt = messages_to_prompt(messages)
    responses: list[LLMResponse] = await api(
        model_id=model, prompt=prompt, temperature=0, max_tokens=1, logprobs=5,
    )
    response = responses[0]
    score = expected_score_from_logprobs(response)

    return {
        "messages": messages,
        "response": response.completion,
        "score": score,
        "tag": tag,
    }


# ---------------------------------------------------------------------------
# 3. Multi-criterion scoring judge — parses <criterion>score</criterion> tags
# ---------------------------------------------------------------------------

TAG_PATTERN = re.compile(r"<(\w+)>\s*(\d+)\s*</\1>")


async def multi_criterion_scoring_judge(
    api: InferenceAPI,
    model: str,
    messages: list[dict[str, str]],
    criteria: list[str],
    tag: str = "",
) -> dict:
    """Multi-criterion judge. Parses XML-style <criterion>score</criterion> tags."""
    prompt = messages_to_prompt(messages)
    responses: list[LLMResponse] = await api(
        model_id=model, prompt=prompt, temperature=0,
    )
    raw = responses[0].completion

    # Parse all <tag>score</tag> pairs
    found = {m.group(1).lower(): int(m.group(2)) for m in TAG_PATTERN.finditer(raw)}

    # Extract scores for requested criteria
    scores = {}
    for criterion in criteria:
        key = criterion.lower()
        scores[criterion] = found.get(key)

    return {
        "messages": messages,
        "response": raw,
        "scores": scores,
        "tag": tag,
    }


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

async def main():
    from pathlib import Path
    from dotenv import load_dotenv

    load_dotenv()

    api = InferenceAPI(
        anthropic_num_threads=10,
        openai_num_threads=10,
        cache_dir=Path("output/cache"),
    )

    pirate_response = "Arrr, the capital o' France be Paris, matey!"
    normal_response = "The capital of France is Paris."

    # --- Binary logit judge ---
    print("=== Binary Logit Judge ===")
    for resp in [pirate_response, normal_response]:
        result = await binary_logit_judge(
            api=api,
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "You are an LLM classifying whether responses are pirate-like."},
                {"role": "user", "content": (
                    "Is this response pirate-like? Respond with ONLY 'yes' or 'no'.\n\n"
                    f"Response: {resp}"
                )},
            ],
            tag="pirate",
        )
        print(f"  {resp[:50]}... -> {result['classification']} (logit={result['logit']})")

    # --- Scoring logit judge ---
    print("\n=== Scoring Logit Judge ===")
    for resp in [pirate_response, normal_response]:
        result = await scoring_logit_judge(
            api=api,
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "You are an LLM scoring how pirate-like a response is."},
                {"role": "user", "content": (
                    "Rate how pirate-like this response is on a scale of 1-5. "
                    "Respond with ONLY a single number.\n\n"
                    f"Response: {resp}"
                )},
            ],
            tag="pirate_score",
        )
        print(f"  {resp[:50]}... -> score={result['score']}")

    # --- Multi-criterion scoring judge ---
    print("\n=== Multi-Criterion Scoring Judge ===")
    criteria = ["pirate_speak", "helpfulness", "accuracy"]
    for resp in [pirate_response, normal_response]:
        result = await multi_criterion_scoring_judge(
            api=api,
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "You are an LLM evaluating responses on multiple criteria."},
                {"role": "user", "content": (
                    "Rate the following response on each criterion (1-5). "
                    "Output your scores in XML tags like <criterion>score</criterion>.\n"
                    "\n"
                    f"Criteria: {', '.join(criteria)}\n"
                    "\n"
                    f"Response: {resp}"
                )},
            ],
            criteria=criteria,
            tag="multi",
        )
        print(f"  {resp[:50]}... -> {result['scores']}")


if __name__ == "__main__":
    asyncio.run(main())
