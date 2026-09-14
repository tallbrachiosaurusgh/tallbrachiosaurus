
from tallbrachiosaurus._core import hello_from_bin
from tallbrachiosaurus.differential.discrete import diff


def hello() -> str:
    return hello_from_bin()


__all__ = ["hello", "diff"]
