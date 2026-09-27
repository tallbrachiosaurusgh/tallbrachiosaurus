import math
import secrets


def uniform(a: float = 0.0, b: float = 1.0) -> float:
    """Cryptographically secure uniform sample."""
    # 53 random bits gives 53-bit precision double
    u = secrets.randbits(53) / (1 << 53)  # in [0, 1)
    return a + (b - a) * u


def exponentialdist(lam: float) -> float:
    if lam <= 0:
        raise ValueError("lambda must be positive")

    # ln(0) is undefined, and uniform() can return exactly 0.0 because its
    # range is [0, 1). Resample in that case rather than returning infinity.
    u = uniform()
    while u == 0.0:
        u = uniform()

    return -math.log(u) / lam


if __name__ == "__main__":
    print("uniform samples:", [round(uniform(), 4) for _ in range(5)])
    print("exponential samples (lambda=2):",
          [round(exponentialdist(2.0), 4) for _ in range(5)])

    n = 100000
    lam = 2.0
    samples = [exponentialdist(lam) for _ in range(n)]
    mean = sum(samples) / n
    print(f"\nSample mean:      {mean:.4f}")
    print(f"Theoretical mean: {1/lam:.4f}")