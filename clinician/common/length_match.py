"""
clinician/common/length_match.py -- cut client outputs to human lengths before rating.

Llama talks for longer than people do (tat_1_full: median 229 words against a
human median of 125), and raters score longer text as more disordered. So
before any rater sees an output, it is cut to a length drawn from the human
word-count distribution, at the last sentence end before that length.

Generation is never capped; this is done on the stored text. A cap only stops
sampling -- the words before it are the same either way -- so cutting here
loses nothing, and the full output stays on record.

Each row's target is drawn from a hash of its row_id, so adding rows to a run
never changes the targets of rows already there.
"""

import bisect
import hashlib
import re

import numpy as np

WORD = re.compile(r"\S+")
# a sentence end: . ! or ? plus any closing quotes/brackets, before space or end
SENT_END = re.compile(r"[.!?][\"')\]”’]*(?=\s|$)")


def squash(text):
    """One line, single spaces -- the human transcripts have no line breaks."""
    return " ".join(str(text).split())


def trim_to(text, target, min_frac=0.5):
    """Cut `text` to about `target` words at the sentence end NEAREST the
    target, before or after it -- cutting only before would make every rated
    text shorter than the human one it was matched to. Returns (text,
    n_words, how) with how in none | sentence | hard. A sentence end is used
    only within min_frac of the target (0.5 -> 50%..150%); otherwise (no
    punctuation -- typical of degenerate output) the cut is at the word."""
    words = list(WORD.finditer(text))
    if len(words) <= target:
        return text, len(words), "none"
    starts = [w.start() for w in words]
    best = None
    for m in SENT_END.finditer(text):
        k = bisect.bisect_left(starts, m.end())          # words kept by cutting here
        if k > target * (1 + min_frac):
            break
        if k >= max(1, target * (1 - min_frac)) and (best is None or abs(k - target) < abs(best[1] - target)):
            best = (m.end(), k)
    if best:
        return text[:best[0]].rstrip(), best[1], "sentence"
    return text[:words[target - 1].end()], target, "hard"


def row_target(row_id, human_words, seed=0):
    """A human word count for this row, fixed by its row_id."""
    h = int(hashlib.sha256(f"{seed}|{row_id}".encode()).hexdigest()[:12], 16)
    return int(human_words[h % len(human_words)])


def match(df, human_words, text_col="output", id_col="row_id", seed=0, min_frac=0.5):
    """Add text_rated / n_words_rated / trim_target / trim to a copy of df."""
    human = np.sort(np.asarray(human_words, dtype=int))
    human = human[human > 0]
    out = df.copy()
    texts, counts, targets, hows = [], [], [], []
    for rid, txt in zip(out[id_col], out[text_col].fillna("").astype(str)):
        t = row_target(rid, human, seed)
        kept, n, how = trim_to(squash(txt), t, min_frac)
        texts.append(kept); counts.append(n); targets.append(t); hows.append(how)
    out["text_rated"], out["n_words_rated"] = texts, counts
    out["trim_target"], out["trim"] = targets, hows
    return out
