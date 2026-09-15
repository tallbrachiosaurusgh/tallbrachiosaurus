import torch
TOL = 1e-10


def rowswap(M, source, target):
    # Work on a copy so the caller's matrix is left untouched.
    result = M.clone()

    if source == target:
        return result

    saved = result[source].clone()
    result[source] = result[target]
    result[target] = saved

    return result


def rowscale(M, source, factor):
    result = M.clone()

    if factor == 0:
        raise ValueError("Scaling factor must be non-zero")

    result[source] = result[source] * float(factor)

    return result


def rowreplacement(M, first, second, j, k):
    result = M.clone()

    if first == second:
        raise ValueError("first and second must be different rows")

    scaled_first = rowscale(result, first, j)[first]
    scaled_second = rowscale(result, second, k)[second]

    result[second] = scaled_first + scaled_second

    return result


def rref(M):
    result = M.clone().float()
    n_rows, n_cols = result.shape

    pivot_row = 0

    for col in range(n_cols):
        # Every row already has a pivot; nothing left to do.
        if pivot_row >= n_rows:
            break

        candidates = torch.abs(result[pivot_row:, col])
        best_offset = int(torch.argmax(candidates).item())
        best_value = float(candidates[best_offset].item())

        if best_value < TOL:
            continue

        pivot_index = pivot_row + best_offset

        if pivot_index != pivot_row:
            result = rowswap(result, pivot_index, pivot_row)

        pivot_value = float(result[pivot_row, col].item())
        result = rowscale(result, pivot_row, 1.0 / pivot_value)

        for row in range(pivot_row + 1, n_rows):
            entry = float(result[row, col].item())
            if abs(entry) > TOL:
                result = rowreplacement(result, pivot_row, row, -entry, 1.0)

        pivot_row += 1

    return result

if __name__ == "__main__":
    A = torch.tensor([[1., 3., 0., 0., 3.],
                      [0., 0., 1., 0., 9.],
                      [0., 0., 0., 1., -4.]])

    print("Original:")
    print(A)

    step1 = rowswap(A, 0, 1)
    print("\nAfter rowswap(A, 0, 1):")
    print(step1)

    step2 = rowscale(step1, 0, 1/3)
    print("\nAfter rowscale(step1, 0, 1/3):")
    print(step2)

    step3 = rowreplacement(step2, 0, 2, -3, 1)
    print("\nAfter rowreplacement(step2, 0, 2, -3, 1):")
    print(step3)

    print("\nOriginal is unchanged:")
    print(A)

    print("\nrref(A):")
    print(rref(A))


    B = torch.tensor([[0., 2., 1., -1.],
                      [0., 4., 2., -2.],
                      [3., 1., 0.,  5.]])
    print("\nB:")
    print(B)
    print("\nrref(B):")
    print(rref(B))

