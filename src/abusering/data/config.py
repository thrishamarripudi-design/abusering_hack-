"""Dataset generation configuration.

The full spec (Sec. 3) asks for 100k customers / 20k merchants / 500k
transactions / ~100 rings. That is achievable but slow in a sandboxed,
single-process environment. We default to a smaller but *proportionally
faithful* "demo" scale and expose a "full" profile that reproduces the
spec's exact target numbers. Both are deterministic given a seed.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class GenerationConfig:
    seed: int = 42
    profile: str = "demo"  # "demo" | "full" | "stress"

    n_customers: int = 8_000
    n_merchants: int = 1_500
    n_transactions: int = 40_000
    n_rings: int = 40
    target_abuse_rate: float = 0.06  # fraction of transactions that are abuse

    start_date: str = "2025-01-01"
    end_date: str = "2025-07-01"

    # relative weights across the 6 ring mechanisms (Sec. 6 A-F)
    ring_type_weights: dict = field(
        default_factory=lambda: {
            "shared_device": 0.22,
            "shared_instrument": 0.20,
            "coordinated_velocity": 0.20,
            "dispute_abuse": 0.15,
            "coordinated_creation": 0.13,
            "hybrid": 0.10,
        }
    )

    def dataset_version(self) -> str:
        return f"{self.profile}-seed{self.seed}-n{self.n_transactions}"


PROFILES: dict[str, GenerationConfig] = {
    "demo": GenerationConfig(
        seed=42,
        profile="demo",
        n_customers=8_000,
        n_merchants=1_500,
        n_transactions=40_000,
        n_rings=40,
        target_abuse_rate=0.06,
    ),
    "full": GenerationConfig(
        seed=42,
        profile="full",
        n_customers=100_000,
        n_merchants=20_000,
        n_transactions=500_000,
        n_rings=100,
        target_abuse_rate=0.06,
    ),
    "stress": GenerationConfig(
        seed=100,
        profile="stress",
        n_customers=20_000,
        n_merchants=4_000,
        n_transactions=80_000,
        n_rings=15,
        target_abuse_rate=0.015,
    ),
}


def get_config(profile: str, seed: int | None = None) -> GenerationConfig:
    if profile not in PROFILES:
        raise ValueError(f"Unknown profile '{profile}'. Options: {list(PROFILES)}")
    cfg = PROFILES[profile]
    if seed is not None:
        cfg = GenerationConfig(**{**cfg.__dict__, "seed": seed})
    return cfg
