import numpy as np

from generals.core.action import Action, compute_valid_move_mask
from generals.core.config import DIRECTIONS
from generals.core.observation import Observation
from generals.core.comabat_utils import should_attack, predict_combat_outcome
from generals.core.channels import UNIT_TYPES

from .agent import Agent


class ExpanderAgent(Agent):
    def __init__(self, id: str = "Expander", win_threshold: float = 0.6):
        super().__init__(id)
        self.win_threshold = win_threshold

    def act(self, observation: Observation) -> Action:
        """
        Heuristically selects a valid (expanding) action.
        Prioritizes capturing opponent and then neutral cells,
        using combat outcome prediction to make intelligent decisions.
        """
        mask = compute_valid_move_mask(observation)
        valid_moves = np.argwhere(mask == 1)

        # Skip the turn if there are no valid moves.
        if len(valid_moves) == 0:
            return Action(to_pass=True)

        unit_arrays = [observation.cavalry, observation.infantry, observation.archers, observation.siege]
        opponent_mask = observation.opponent_cells
        neutral_mask = observation.neutral_cells

        # Store moves with their predicted win probabilities
        opponent_moves = []
        neutral_moves = []

        for move_idx, move in enumerate(valid_moves):
            orig_row, orig_col, direction, unit_type_idx = move

            # Get the destination position
            row_offset, col_offset = DIRECTIONS[direction].value
            dest_row, dest_col = (orig_row + row_offset, orig_col + col_offset)

            orig_pos = (orig_row, orig_col)
            dest_pos = (dest_row, dest_col)

            # Only consider moves with enough units
            unit_array = unit_arrays[unit_type_idx]
            if unit_array[orig_row, orig_col] <= 1:
                continue

            # For opponent cells, use combat prediction to assess win probability
            if opponent_mask[dest_row, dest_col]:
                # should_attack expects unit_type_idx as an integer
                if should_attack(observation, orig_pos, dest_pos, unit_type_idx, self.win_threshold):
                    # predict_combat_outcome expects unit_type as a string
                    unit_type_str = UNIT_TYPES[unit_type_idx]
                    # Predict win probability
                    win_prob, _ = predict_combat_outcome(
                        observation, orig_pos, dest_pos, unit_type_str, unit_array[orig_row, orig_col] - 1
                    )
                    opponent_moves.append((move, win_prob))

            # For neutral cells, we'll also evaluate the likelihood of success
            elif neutral_mask[dest_row, dest_col]:
                # Neutral cells are usually easier to capture
                # predict_combat_outcome expects unit_type as a string
                unit_type_str = UNIT_TYPES[unit_type_idx]
                win_prob, _ = predict_combat_outcome(
                    observation, orig_pos, dest_pos, unit_type_str, unit_array[orig_row, orig_col] - 1
                )
                if win_prob >= self.win_threshold:
                    neutral_moves.append((move, win_prob))

        # Prioritize opponent moves, then neutral moves
        selected_move = None

        if opponent_moves:
            # Sort by win probability (descending)
            opponent_moves.sort(key=lambda x: x[1], reverse=True)
            selected_move = opponent_moves[0][0]
        elif neutral_moves:
            # Sort by win probability (descending)
            neutral_moves.sort(key=lambda x: x[1], reverse=True)
            selected_move = neutral_moves[0][0]
        else:
            # Otherwise, select a random valid action
            move_index = np.random.choice(len(valid_moves))
            selected_move = valid_moves[move_index]

        action = Action(
            to_pass=False,
            row=selected_move[0],
            col=selected_move[1],
            direction=selected_move[2],
            unit_type_idx=selected_move[3],
            to_split=False,
        )

        return action

    def reset(self):
        pass
