"""Code-generated training data: natural date phrase -> ISO 8601.

The whole point of a nano model is to pick a task narrow enough that ~1M
parameters can actually *nail* it, and structured enough that the data needs no
scraping and no labelling: we sample the answer first, then render it in many
natural surface forms, so every (prompt, target) pair is correct by construction.
That is strictly better than asking a big model to produce data — no verification,
no cost, unlimited, and the label is the ground truth, not a guess.

Each example is one line::

    2024-03-10 | next friday => 2024-03-15

The reference date (``today``) is given to the model as ISO at the start of the
prompt, so relative phrases ("tomorrow", "in 3 days", "next friday") are
*computable* from the input alone. The prompt is everything up to and including
``" => "``; the target is the 10-character ISO date that follows.

Two design rules keep the task honest:

* **Unambiguous inputs only.** We never emit "12/6/2023" (June or December?).
  Numeric months are always words ("June", "Jun"); the only all-numeric form is ISO.
* **Absolute forms use an INDEPENDENT reference date.** If an absolute phrase's
  date equalled the prompt's reference date, the model could "solve" the form by
  copying the ISO prefix instead of reading the phrase — a degenerate shortcut that
  scores ~100% while learning nothing. Decoupling them forces genuine parsing.
"""

from __future__ import annotations

import calendar
from datetime import date, timedelta
from typing import Callable

import numpy as np

# Cross-entropy sentinel: y positions set to this are skipped by the loss. We mask
# the whole prompt so the model is only ever scored on producing the ISO target.
IGNORE_INDEX = -100

_MONTHS = ["January", "February", "March", "April", "May", "June",
           "July", "August", "September", "October", "November", "December"]
_ABBR = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
         "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_ORDINALS = {1: "st", 2: "nd", 3: "rd", 21: "st", 22: "nd", 23: "rd", 31: "st"}

# Reference dates are drawn from this window: wide enough that the model must learn
# the rule (not memorise a year), bounded so the arithmetic stays in range.
_MIN_ORD = date(2015, 1, 1).toordinal()
_MAX_ORD = date(2035, 12, 31).toordinal()


def _ord(day: int) -> str:
    return f"{day}{_ORDINALS.get(day, 'th')}"


def _add_months(d: date, n: int) -> date:
    """d shifted by n calendar months, clamping the day (Jan 31 + 1mo -> Feb 28)."""
    m = d.month - 1 + n
    y = d.year + m // 12
    m = m % 12 + 1
    return date(y, m, min(d.day, calendar.monthrange(y, m)[1]))


def _next_weekday(d: date, wd: int) -> date:
    return d + timedelta(days=((wd - d.weekday()) % 7) or 7)


def _last_weekday(d: date, wd: int) -> date:
    return d - timedelta(days=((d.weekday() - wd) % 7) or 7)


def _rand_date(r) -> date:
    return date.fromordinal(int(r.integers(_MIN_ORD, _MAX_ORD + 1)))


# Each renderer maps (rng, today) -> (phrase, answer date). Absolute forms sample
# their OWN answer independent of `today` (see module docstring).
def _renderers() -> list[Callable]:
    def abs_iso(r, t):
        a = _rand_date(r); return a.isoformat(), a

    def abs_long(r, t):
        a = _rand_date(r); return f"{_MONTHS[a.month-1]} {a.day}, {a.year}", a

    def abs_abbr(r, t):
        a = _rand_date(r); return f"{_ABBR[a.month-1]} {a.day} {a.year}", a

    def abs_dmy(r, t):
        a = _rand_date(r); return f"{a.day} {_MONTHS[a.month-1]} {a.year}", a

    def abs_ordinal(r, t):
        a = _rand_date(r); return f"the {_ord(a.day)} of {_MONTHS[a.month-1]} {a.year}", a

    def rel_today(r, t):     return "today", t
    def rel_tomorrow(r, t):  return "tomorrow", t + timedelta(days=1)
    def rel_yesterday(r, t): return "yesterday", t - timedelta(days=1)

    def rel_in_days(r, t):
        n = int(r.integers(2, 31)); return f"in {n} days", t + timedelta(days=n)

    def rel_days_ago(r, t):
        n = int(r.integers(2, 31)); return f"{n} days ago", t - timedelta(days=n)

    def rel_in_weeks(r, t):
        n = int(r.integers(1, 9)); return f"in {n} weeks", t + timedelta(weeks=n)

    def rel_next_week(r, t): return "next week", t + timedelta(weeks=1)
    def rel_last_week(r, t): return "last week", t - timedelta(weeks=1)

    def rel_next_weekday(r, t):
        wd = int(r.integers(0, 7)); return f"next {_WEEKDAYS[wd]}", _next_weekday(t, wd)

    def rel_last_weekday(r, t):
        wd = int(r.integers(0, 7)); return f"last {_WEEKDAYS[wd]}", _last_weekday(t, wd)

    def rel_in_months(r, t):
        n = int(r.integers(1, 13)); return f"in {n} months", _add_months(t, n)

    def rel_next_month(r, t): return "next month", _add_months(t, 1)

    return [abs_iso, abs_long, abs_abbr, abs_dmy, abs_ordinal,
            rel_today, rel_tomorrow, rel_yesterday, rel_in_days, rel_days_ago,
            rel_in_weeks, rel_next_week, rel_last_week, rel_next_weekday,
            rel_last_weekday, rel_in_months, rel_next_month]


def date_pairs(seed: int, n: int) -> list[tuple[str, str]]:
    """`n` deterministic (prompt, iso-target) pairs from `seed`."""
    rng = np.random.default_rng(seed)
    renderers = _renderers()
    out = []
    for _ in range(n):
        today = date.fromordinal(int(rng.integers(_MIN_ORD, _MAX_ORD + 1)))
        phrase, answer = renderers[int(rng.integers(len(renderers)))](rng, today)
        out.append((f"{today.isoformat()} | {phrase} => ", answer.isoformat()))
    return out


def encode_pair(prompt: str, target: str, seq_len: int, pad_id: int = 0):
    """Byte-encode prompt+target into next-token (x, y) with the prompt masked.

    y positions over the prompt and padding are IGNORE_INDEX, so cross-entropy only
    scores the 10-char ISO target.
    """
    p = list(prompt.encode("utf-8"))
    t = list(target.encode("utf-8"))
    full = p + t
    plen = len(p)
    over = len(full) - (seq_len + 1)
    if over > 0:  # left-truncate the prompt so every target token survives
        cut = min(over, plen - 1)
        full = full[cut:]
        plen -= cut
    x, y = full[:-1], full[1:]
    mask_upto = plen - 1
    y = [IGNORE_INDEX] * mask_upto + y[mask_upto:]
    x = (x + [pad_id] * seq_len)[:seq_len]
    y = (y + [IGNORE_INDEX] * seq_len)[:seq_len]
    return x, y


if __name__ == "__main__":
    for prompt, target in date_pairs(0, 12):
        print(f"{prompt}{target}")
