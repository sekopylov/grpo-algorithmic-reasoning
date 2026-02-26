import re


def _last_r0(text: str):
    matches = list(re.finditer(r"R0\s*=\s*(-?\d+)", text, flags=re.IGNORECASE))
    if not matches:
        return None
    return f"R0={int(matches[-1].group(1))}"


def _extract_answer(solution_str: str):
    tag_match = re.search(r"<answer>(.*?)(</answer>|$)", solution_str, flags=re.IGNORECASE | re.DOTALL)
    if tag_match:
        parsed = _last_r0(tag_match.group(1).strip())
        if parsed is not None:
            return parsed

    after_think = re.split(r"</think>", solution_str, maxsplit=1, flags=re.IGNORECASE)
    if len(after_think) != 2:
        return None

    return _last_r0(after_think[1].strip())


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    parsed = _extract_answer(solution_str)
    return 1.0 if parsed == ground_truth else 0.0
