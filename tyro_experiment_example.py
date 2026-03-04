"""Minimal tyro experiment config example."""

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import tyro


@dataclass
class ModelConfig:
    """Model configuration."""
    provider: str = "openai"        # API provider
    model: str = "gpt-4o"           # Model name
    temperature: float = 1.0        # Sampling temperature
    max_tokens: int = 1024          # Max tokens per completion


@dataclass
class ExperimentConfig:
    """Run an experiment with configurable model and eval settings."""
    model: ModelConfig = ModelConfig()
    dataset_path: str = "data/dataset.json"   # Path to input dataset
    output_dir: str | None = None             # Output dir (default: timestamped)
    num_samples: int = 10                     # Samples per condition
    max_concurrent: int = 32                  # Max concurrent API calls
    verbose: bool = False                     # Verbose logging


def run_experiment(config: ExperimentConfig) -> dict:
    """Mock experiment: 'generates' responses and 'scores' them."""
    # Setup output dir
    if config.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = Path("output") / timestamp
    else:
        out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(asdict(config), indent=2))

    # Mock generation + scoring
    results = []
    for i in range(config.num_samples):
        results.append({
            "sample_idx": i,
            "model": config.model.model,
            "response": f"Mock response {i} (t={config.model.temperature})",
            "score": 1.0 - (i % 3) * 0.2,
        })

    # Save results + metrics
    (out / "results.json").write_text(json.dumps(results, indent=2))
    avg_score = sum(r["score"] for r in results) / len(results)
    metrics = {"avg_score": avg_score, "num_samples": len(results)}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print(f"Wrote {len(results)} results to {out}")
    print(f"Avg score: {avg_score:.3f}")
    return metrics


def main() -> None:
    config = tyro.cli(ExperimentConfig)
    run_experiment(config)


if __name__ == "__main__":
    main()
