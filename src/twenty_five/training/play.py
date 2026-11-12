"""Readable terminal game for humans playing saved checkpoints and baselines."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from ..cards import Card
from ..engine import Phase, TwentyFiveGame
from ..policies import FrozenCheckpointPolicy, Policy, PolicyView, make_policy


def format_table(game: TwentyFiveGame, human_seat: int) -> str:
    current = " ".join(f"P{seat}:{card}" for seat, card in game.current_trick) or "(empty)"
    hand = " ".join(str(card) for card in sorted(game.hands[human_seat], key=lambda card: card.index))
    lines = [
        "",
        f"Round {game.round_number} | Scores: " + "  ".join(f"P{seat}={score}" for seat, score in enumerate(game.scores)),
        f"Trump: {game.trump.name.title()} ({game.turned_card}) | Dealer: P{game.dealer} | Leader: P{game.leader}",
        f"Current trick: {current}",
        f"Your hand: {hand}",
    ]
    if game.trick_winners:
        lines.append(f"Last trick winner: P{game.trick_winners[-1]}")
    if game.phase is Phase.ROB and game.current_player == human_seat:
        lines.append("You must rob: choose one card to discard.")
    return "\n".join(lines)


def _human_action(game: TwentyFiveGame, seat: int) -> int:
    legal = sorted(game.legal_cards(seat), key=lambda card: card.index)
    while True:
        print(format_table(game, seat))
        for number, card in enumerate(legal, 1):
            print(f"  {number}. {card}")
        choice = input(f"Choose a card (1-{len(legal)}, card name, or q): ").strip().upper()
        if choice in {"Q", "QUIT", "EXIT"}:
            raise KeyboardInterrupt
        if choice.isdigit():
            value = int(choice)
            if 1 <= value <= len(legal):
                return legal[value - 1].index
            if any(card.index == value for card in legal):
                return value
        for card in legal:
            if str(card).upper() == choice:
                return card.index
        print("That is not one of the legal cards.")


def play_human_game(
    players: int,
    checkpoint: Path | None,
    opponent: str,
    human_seat: int,
    seed: int,
    dealer: int,
    save: Path | None = None,
) -> TwentyFiveGame:
    if not 0 <= human_seat < players:
        raise ValueError("human seat outside player range")
    ai: Policy = FrozenCheckpointPolicy(checkpoint) if checkpoint else make_policy(opponent)
    game = TwentyFiveGame(players, seed=seed, initial_dealer=dealer, debug=True)
    initial = game.to_dict()
    actions: list[dict[str, int | str]] = []
    rng = random.Random(seed ^ 0xBEEF)
    try:
        while not game.terminal:
            seat = game.current_player
            if seat == human_seat:
                action = _human_action(game, seat)
                actor = "human"
            else:
                view = PolicyView.from_game(game, seat)
                action = ai.act(view, rng)
                actor = ai.name
                if game.phase is Phase.ROB:
                    print(f'P{seat} robs and discards one card face down')
                else:
                    print(f"P{seat} plays {Card.from_index(action)}")
            actions.append({"seat": seat, "action": action, "actor": actor})
            game.step(seat, action)
    except KeyboardInterrupt:
        print("\nGame stopped.")
    else:
        print(format_table(game, human_seat))
        print(f"\nWinner: P{game.winner} | Final scores: {game.scores}")
    finally:
        if save is not None:
            save.parent.mkdir(parents=True, exist_ok=True)
            save.write_text(
                json.dumps(
                    {
                        "format_version": 1,
                        "initial": initial,
                        "actions": actions,
                        "checkpoint": str(checkpoint) if checkpoint else None,
                        "opponent": opponent,
                        "human_seat": human_seat,
                    },
                    indent=2,
                )
                + "\n"
            )
    return game


def replay_game(path: Path, pause: bool = False) -> TwentyFiveGame:
    data = json.loads(path.read_text())
    game = TwentyFiveGame.from_dict(data["initial"])
    human_seat = int(data.get("human_seat", 0))
    print(format_table(game, human_seat))
    for row in data["actions"]:
        if isinstance(row, dict):
            seat, action = int(row["seat"]), int(row["action"])
        else:
            seat, action = map(int, row)
        if seat != game.current_player:
            raise ValueError(f"replay expected P{game.current_player}, found action for P{seat}")
        print(f"P{seat} plays {Card.from_index(action)}")
        game.step(seat, action)
        if pause:
            input("Press Enter for the next action...")
    print(format_table(game, human_seat))
    print(f"\nReplay complete | Winner: P{game.winner} | Scores: {game.scores}")
    return game


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", type=Path, help="play back an exact JSON game record")
    parser.add_argument("--checkpoint", type=Path, help="schema-versioned .pt or .json policy used by every AI seat")
    parser.add_argument("--opponent", choices=["random", "lowest", "highest", "greedy", "conservative", "counting"], default="counting")
    parser.add_argument("--players", type=int, default=4)
    parser.add_argument("--human-seat", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dealer", type=int, default=0)
    parser.add_argument("--save", type=Path, help="write an exactly replayable JSON game record")
    args = parser.parse_args()
    if args.replay:
        replay_game(args.replay)
        return
    play_human_game(args.players, args.checkpoint, args.opponent, args.human_seat, args.seed, args.dealer, args.save)


if __name__ == "__main__":
    main()
