def diff(t, x):
    # Reject mismatched inputs before computing anything
    if len(t) != len(x):
        raise ValueError(
            f"Length mismatch"
        )

    # Need at least two samples to form a difference
    if len(t) < 2:
        raise ValueError("Need at least two samples to compute a derivative")

    v = []

    # Start at index 1 since each value depends on its predecessor
    for k in range(1, len(t)):
        dt = t[k] - t[k - 1]

        # Guard against division by zero from duplicate timestamps
        if dt == 0:
            raise ValueError(f"Duplicate time value at index {k}: t={t[k]}")

        dx = x[k] - x[k - 1]
        v.append(dx / dt)

    return v
