# coding=utf-8
"""Smooth BLEU utilities.

Faithful, configurable Smooth BLEU (ported from MOSES/NIST style).
This is intended as a drop-in replacement for code-eval setups that use
`smooth BLEU` for summarization tasks.

Public API:
    compute_smooth_bleu(refs, hyps, n=4, smooth=1, eff_ref_len="shortest",
                        preserve_case=False, nonorm=False) -> float

Notes:
- Returns corpus-level BLEU in range [0, 1].
- `refs` is List[List[str]]: list over examples, each with list of reference strings.
- `hyps` is List[str].
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import math
import re
import xml.sax.saxutils


# --- Normalization (faithful to the reference implementation) ---
_normalize1_patterns = [
    (r"<skipped>", ""),  # strip "skipped" tags
    (r"-\n", ""),  # strip end-of-line hyphenation and join lines
    (r"\n", " "),  # join lines
]
_normalize1 = [(re.compile(pat), rep) for (pat, rep) in _normalize1_patterns]

_normalize2_patterns = [
    (r"([\{-\~\[-\` -\&\(-\+\:-\@\/])", r" \1 "),  # tokenize punctuation (apostrophe omitted)
    (r"([^0-9])([\.,])", r"\1 \2 "),  # period/comma unless preceded by digit
    (r"([\.,])([^0-9])", r" \1 \2"),  # period/comma unless followed by digit
    (r"([0-9])(-)", r"\1 \2 "),  # dash when preceded by a digit
]
_normalize2 = [(re.compile(pat), rep) for (pat, rep) in _normalize2_patterns]


def _split_tokens_like_ref(
    s: str,
    *,
    preserve_case: bool = False,
    nonorm: bool = False,
) -> List[str]:
    """Normalize + tokenize like the reference script.

    If nonorm=True, simply splits on whitespace (bypass NIST-style pre-processing).
    """

    if nonorm:
        return s.split()

    if not isinstance(s, str):
        s = " ".join(s)

    for (pattern, repl) in _normalize1:
        s = pattern.sub(repl, s)

    s = xml.sax.saxutils.unescape(s, {"&quot;": '"'})

    s = f" {s} "
    if not preserve_case:
        s = s.lower()

    for (pattern, repl) in _normalize2:
        s = pattern.sub(repl, s)

    return s.split()


def _count_ngrams(words: List[str], n: int = 4) -> Dict[Tuple[str, ...], int]:
    counts: Dict[Tuple[str, ...], int] = {}
    for k in range(1, n + 1):
        for i in range(len(words) - k + 1):
            ng = tuple(words[i : i + k])
            counts[ng] = counts.get(ng, 0) + 1
    return counts


def _cook_refs(
    refs: List[str],
    *,
    n: int = 4,
    preserve_case: bool = False,
    nonorm: bool = False,
) -> Tuple[List[int], Dict[Tuple[str, ...], int]]:
    """Return (reference_lengths, max_ref_ngram_counts)."""

    refs_tok = [_split_tokens_like_ref(r, preserve_case=preserve_case, nonorm=nonorm) for r in refs]
    maxcounts: Dict[Tuple[str, ...], int] = {}
    for ref in refs_tok:
        counts = _count_ngrams(ref, n=n)
        for ng, c in counts.items():
            if c > maxcounts.get(ng, 0):
                maxcounts[ng] = c

    return [len(ref) for ref in refs_tok], maxcounts


def _cook_test(
    hyp: str,
    cooked_refs: Tuple[List[int], Dict[Tuple[str, ...], int]],
    *,
    n: int = 4,
    eff_ref_len: str = "shortest",
    preserve_case: bool = False,
    nonorm: bool = False,
) -> Dict[str, object]:
    reflens, refmaxcounts = cooked_refs
    hyp_tok = _split_tokens_like_ref(hyp, preserve_case=preserve_case, nonorm=nonorm)

    result: Dict[str, object] = {}
    testlen = len(hyp_tok)
    result["testlen"] = testlen

    if eff_ref_len == "shortest":
        result["reflen"] = min(reflens)
    elif eff_ref_len == "average":
        result["reflen"] = float(sum(reflens)) / max(1, len(reflens))
    elif eff_ref_len == "closest":
        best_ref = reflens[0]
        min_diff = abs(reflens[0] - testlen)
        for rl in reflens[1:]:
            d = abs(rl - testlen)
            if d < min_diff:
                min_diff = d
                best_ref = rl
        result["reflen"] = best_ref
    else:
        raise ValueError("eff_ref_len must be one of {'shortest','average','closest'}")

    result["guess"] = [max(testlen - k + 1, 0) for k in range(1, n + 1)]

    result["correct"] = [0] * n
    h_counts = _count_ngrams(hyp_tok, n=n)
    for ng, c in h_counts.items():
        result["correct"][len(ng) - 1] += min(refmaxcounts.get(ng, 0), c)

    return result


def _score_cooked(allcomps: Iterable[Dict[str, object]], *, n: int = 4, smooth: int = 1) -> List[float]:
    total = {"testlen": 0, "reflen": 0, "guess": [0] * n, "correct": [0] * n}
    for comps in allcomps:
        total["testlen"] += int(comps["testlen"])  # type: ignore[arg-type]
        total["reflen"] += int(comps["reflen"])  # type: ignore[arg-type]
        for k in range(n):
            total["guess"][k] += int(total["guess"][k] + 0) * 0 + int(comps["guess"][k])  # type: ignore[index]
            total["correct"][k] += int(comps["correct"][k])  # type: ignore[index]

    logbleu = 0.0
    per_order_logs = []
    for k in range(n):
        correct = total["correct"][k]
        guess = total["guess"][k]
        addsmooth = 1 if (smooth == 1 and k > 0) else 0
        logbleu += math.log(correct + addsmooth + 1e-16) - math.log(guess + addsmooth + 1e-16)
        if guess == 0:
            per_order_logs.append(-1e7)
        else:
            per_order_logs.append(math.log(correct + 1e-16) - math.log(guess + 1e-16))

    logbleu /= float(n)
    per_order_logs.insert(0, logbleu)

    brev_penalty = min(0.0, 1.0 - float(total["reflen"] + 1) / float(total["testlen"] + 1))
    out: List[float] = []
    for i, val in enumerate(per_order_logs):
        if i == 0:
            val += brev_penalty
        out.append(math.exp(val))
    return out


def compute_smooth_bleu(
    refs: List[List[str]],
    hyps: List[str],
    *,
    n: int = 4,
    smooth: int = 1,
    eff_ref_len: str = "shortest",
    preserve_case: bool = False,
    nonorm: bool = False,
) -> float:
    """Compute (faithful) Smooth BLEU for many items.

    Returns corpus-level BLEU in [0, 1].
    """

    assert len(refs) == len(hyps), "refs and hyps must have the same length"

    allc = []
    for rlist, hyp in zip(refs, hyps):
        cooked_refs = _cook_refs(rlist, n=n, preserve_case=preserve_case, nonorm=nonorm)
        cooked_test = _cook_test(
            hyp,
            cooked_refs,
            n=n,
            eff_ref_len=eff_ref_len,
            preserve_case=preserve_case,
            nonorm=nonorm,
        )
        allc.append(cooked_test)

    return _score_cooked(allc, n=n, smooth=smooth)[0]
