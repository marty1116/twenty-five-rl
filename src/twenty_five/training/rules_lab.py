"""Display independently specified expected moves beside actual rules output."""

import argparse
import json
from pathlib import Path

from ..cards import FULL_DECK, Suit
from ..rules import legal_cards


def check_cases(path):
    cards = {str(card): card for card in FULL_DECK}
    results = []
    for case in json.loads(path.read_text(encoding="utf-8")):
        actual = [str(card) for card in legal_cards(
            [cards[name] for name in case["hand"]], Suit[case["trump"]], [(0, cards[case["lead"]])])]
        results.append({**case, "actual": actual, "passed": set(actual) == set(case["legal"])})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", type=Path, nargs="?", default=Path("configs/research/rules_cases.json"))
    args = parser.parse_args()
    results = check_cases(args.cases)
    print(json.dumps(results, indent=2))
    raise SystemExit(0 if results and all(row["passed"] for row in results) else 1)


if __name__ == "__main__":
    main()
