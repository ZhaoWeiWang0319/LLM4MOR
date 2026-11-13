import json
import math
import sys


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: python3 map.py input.json output.json")

    with open(sys.argv[1], "r", encoding="utf-8") as f:
        inp = json.load(f)

    variables = inp.get("variables", {})
    if not isinstance(variables, dict):
        raise ValueError("variables must be a JSON object")

    out = {}
    for name, value in variables.items():
        if not isinstance(name, str):
            raise ValueError("variable name must be a string")
        is_original_x = name.startswith("x[") and name.endswith("]")
        is_original_y = name == "y"
        if not (is_original_x or is_original_y):
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"variable {name} must be a finite number")
        if not math.isfinite(value):
            raise ValueError(f"variable {name} must be finite")
        out[name] = value

    if "y" not in out:
        raise ValueError("missing variable y")

    with open(sys.argv[2], "w", encoding="utf-8") as f:
        json.dump({"variables": out}, f, allow_nan=False)


if __name__ == "__main__":
    main()