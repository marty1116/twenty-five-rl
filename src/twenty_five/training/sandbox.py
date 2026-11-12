"""Interactive, replay, and duplicate-deal policy sandbox."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from ..engine import TwentyFiveGame
from ..evaluation import play_match
from ..observations import PlayerView
from ..policies import make_policy
from ..valuation import discard_value, intrinsic_strength


def run(
    players: int,
    names: list[str],
    seed: int,
    replay: Path | None = None,
    save: Path | None = None,
    debug_hands: bool = False,
    interactive_seat: int | None = None,
) -> None:
    if not names:
        raise ValueError("at least one opponent policy is required")
    if interactive_seat is not None and not 0 <= interactive_seat < players:
        raise ValueError("interactive seat outside player range")
    if replay:
        data = json.loads(replay.read_text(encoding="utf-8"))
        game = TwentyFiveGame.from_dict(data["initial"])
        for row in data["actions"]:
            player, action = (int(row["seat"]), int(row["action"])) if isinstance(row, dict) else map(int, row)
            game.step(player, action)
        print(json.dumps(game.public_state(), indent=2))
        return

    policies = [make_policy(names[index % len(names)]) for index in range(players)]
    game = TwentyFiveGame(players, seed, debug=True)
    initial = game.to_dict()
    actions: list[tuple[int, int]] = []
    rng = random.Random(seed)
    while not game.terminal:
        player = game.current_player
        legal = game.legal_cards()
        values = {
            str(card): {
                "index": card.index,
                "intrinsic": intrinsic_strength(card, game.trump),
                "discard": (
                    discard_value(card, game.hands[player], game.trump)
                    if game.phase.value == "rob"
                    else None
                ),
            }
            for card in legal
        }
        display = {
            "public": game.public_state(),
            "seat": player,
            "hand": [str(card) for card in game.hands[player]],
            "legal_values": values,
            "debug_hands": (
                [[str(card) for card in hand] for hand in game.hands] if debug_hands else None
            ),
        }
        print(json.dumps(display, default=str))
        if interactive_seat == player:
            legal_indices = {card.index for card in legal}
            while True:
                choice = input("card index> ").strip()
                if choice.isdigit() and int(choice) in legal_indices:
                    action = int(choice)
                    break
                print(f"Please enter one of the legal action indices: {sorted(legal_indices)}")
        else:
            action = policies[player].act(PlayerView.from_game(game, player), rng)
        actions.append((player, action))
        game.step(player, action)
    print(f"winner={game.winner} scores={game.scores}")
    if save:
        save.parent.mkdir(parents=True, exist_ok=True)
        save.write_text(json.dumps({"initial": initial, "actions": actions}, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--players", type=int, default=4)
    parser.add_argument("--opponents", nargs="+", default=["random"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--save", type=Path)
    parser.add_argument("--debug-hands", action="store_true")
    parser.add_argument("--interactive-seat", type=int)
    parser.add_argument("--compare", nargs=2, metavar=("POLICY_A", "POLICY_B"))
    args = parser.parse_args()
    if args.compare:
        first, second = map(make_policy, args.compare)
        results = []
        for seat in range(2):
            lineup = [second, second]
            lineup[seat] = first
            game, _ = play_match(lineup, args.seed, dealer=seat)
            results.append({"policy_a_seat": seat, "winner": game.winner, "scores": game.scores})
        print(json.dumps({"duplicate_seed": args.seed, "seat_rotations": results}, indent=2))
        return
    run(
        args.players,
        args.opponents,
        args.seed,
        args.replay,
        args.save,
        args.debug_hands,
        args.interactive_seat,
    )


if __name__ == "__main__":
    main()

