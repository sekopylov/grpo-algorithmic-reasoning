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


def compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    length_alpha=0.03,
    length_ref_chars=8000,
    apply_len_only_on_train=True,
    **kwargs,
):
    parsed = _extract_answer(solution_str)
    if parsed != ground_truth:
        return 0.0

    split = ""
    if isinstance(extra_info, dict):
        split = str(extra_info.get("split", ""))

    if apply_len_only_on_train and split.startswith("val"):
        return 1.0

    length_ratio = min(len(solution_str) / float(length_ref_chars), 1.0)
    score = 1.0 - float(length_alpha) * length_ratio
    if score < 0.0:
        score = 0.0
    return score
