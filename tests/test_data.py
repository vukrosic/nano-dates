"""The dates task: correct labels, correct masking, no answer leak, deterministic.

These guard the things a code-generated task must get right — the ground-truth
label really is the right answer, only the ISO target is supervised, the absolute
forms don't leak the answer through the reference date, and the data is
deterministic so splits are reproducible.

    python -m pytest tests/ -q
"""

from __future__ import annotations

import calendar
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from generate_data import IGNORE_INDEX, date_pairs, encode_pair


def _recompute(phrase: str, today: date):
    if phrase == "today":
        return today
    if phrase == "tomorrow":
        return today + timedelta(days=1)
    if phrase == "yesterday":
        return today - timedelta(days=1)
    if phrase == "next week":
        return today + timedelta(days=7)
    if phrase == "last week":
        return today - timedelta(days=7)
    if phrase.startswith("in ") and phrase.endswith(" days"):
        return today + timedelta(days=int(phrase.split()[1]))
    if phrase.endswith(" days ago"):
        return today - timedelta(days=int(phrase.split()[0]))
    if phrase.startswith("in ") and phrase.endswith(" weeks"):
        return today + timedelta(days=7 * int(phrase.split()[1]))
    return None


def test_labels_are_correct():
    # Every relative-day phrase we can independently recompute must match exactly.
    checked = 0
    for prompt, target in date_pairs(0, 3000):
        today_iso, rest = prompt.split(" | ", 1)
        phrase = rest.rsplit(" => ", 1)[0]
        expect = _recompute(phrase, date.fromisoformat(today_iso))
        if expect is not None:
            assert target == expect.isoformat(), (prompt, target, expect)
            checked += 1
    assert checked > 200


def test_absolute_forms_are_valid_dates():
    for prompt, target in date_pairs(1, 3000):
        d = date.fromisoformat(target)
        assert 2014 <= d.year <= 2037
        assert 1 <= d.month <= 12
        assert 1 <= d.day <= calendar.monthrange(d.year, d.month)[1]


def test_absolute_answer_decoupled_from_today():
    # Regression guard: the original generator set today == the absolute answer, so
    # the model could score ~100% by copying the ISO prefix instead of parsing. For
    # a 4-digit-year phrase, the answer must almost never coincide with today.
    coincide = total = 0
    for prompt, target in date_pairs(5, 4000):
        today_iso, rest = prompt.split(" | ", 1)
        phrase = rest.rsplit(" => ", 1)[0]
        if any(t.isdigit() and len(t) == 4 for t in phrase.replace(",", "").split()):
            total += 1
            coincide += (target == today_iso)
    assert total > 300, total
    assert coincide / total < 0.02, (coincide, total)


def test_only_target_is_supervised():
    # Exactly the 10 ISO characters are supervised; prompt + padding are masked.
    for prompt, target in date_pairs(2, 200):
        _, y = encode_pair(prompt, target, 64)
        assert sum(1 for v in y if v != IGNORE_INDEX) == 10


def test_deterministic():
    assert date_pairs(0, 100) == date_pairs(0, 100)
    assert date_pairs(0, 100) != date_pairs(1, 100)
