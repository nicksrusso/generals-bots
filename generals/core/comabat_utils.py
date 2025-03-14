"""
Utility functions for predicting combat outcomes, to be used by agents.
"""

from generals.core.observation import Observation
from generals.core.channels import COMBAT_EFFECTIVENESS, UNIT_TYPES

import random
import math

WIN_PROB_COEF = 2.85


def predict_combat_outcome(
    observation: Observation,
    attacker_pos: tuple[int, int],
    defender_pos: tuple[int, int],
    attacking_unit_type: str,
    attacking_unit_count: float,
) -> tuple[float, dict[str, float]]:
    """
    Predicts the outcome of combat between attacking and defending positions.

    Args:
        observation: The current observation
        attacker_pos: (row, col) of the attacking position
        defender_pos: (row, col) of the defending position
        attacking_unit_type: Type of unit being used for attack
        attacking_unit_count: Number of attacking units

    Returns:
        tuple: (win_probability, remaining_units)
            win_probability: Estimated probability of attacker winning (0.0-1.0)
            remaining_units: Estimated remaining units if attacker wins
    """
    # Get defender's army composition
    defender_units = {}
    a_row, a_col = attacker_pos
    d_row, d_col = defender_pos

    # If the defender position is visible (not in fog)
    if not observation.fog_cells[d_row, d_col]:
        # Get actual unit counts at defender position
        defender_units = {
            "cavalry": observation.cavalry[d_row, d_col],
            "infantry": observation.infantry[d_row, d_col],
            "archers": observation.archers[d_row, d_col],
            "siege": observation.siege[d_row, d_col],
        }
    else:
        # If defender is in fog, make an estimation based on visible armies
        # This is a simple estimate - in a real agent you might use more sophisticated heuristics
        total_defending = observation.armies[d_row, d_col]

        # Default to equal distribution if we can't see
        for unit_type in UNIT_TYPES:
            defender_units[unit_type] = total_defending / len(UNIT_TYPES)

    # Calculate combat power
    attacker_power = 0
    for def_type, def_count in defender_units.items():
        effectiveness = COMBAT_EFFECTIVENESS[attacking_unit_type][def_type]
        attacker_power += attacking_unit_count * effectiveness * def_count

    defender_power = 0
    for def_type, def_count in defender_units.items():
        effectiveness = COMBAT_EFFECTIVENESS[def_type][attacking_unit_type]
        defender_power += def_count * effectiveness * attacking_unit_count

    # Calculate win probability (simplified approximation)
    # A power ratio of 1.0 means about 50% chance to win
    # Higher ratios increase probability, lower ratios decrease it
    strength_ratio = attacker_power / defender_power
    win_probability = strength_ratio**WIN_PROB_COEF / (strength_ratio**WIN_PROB_COEF + 1)  # Probability attacker wins

    # Calculate expected loss ratio
    expected_loss_ratio = calculate_expected_loss_ratio(strength_ratio)

    return win_probability, expected_loss_ratio


def calculate_expected_loss_ratio(strength_ratio):
    """
    Calculate the expected proportion of attacking units that would be lost in combat.
    Used by agents for planning/decision-making.

    Args:
        strength_ratio: Ratio of attacker strength to defender strength

    Returns:
        float: Expected proportion of attacking units that would be lost (0.2-0.8)
    """
    # Base deterministic loss calculation (same as original)
    expected_loss_ratio = min(0.8, 1 - (strength_ratio - 1.0) * 0.2)

    # Ensure minimum losses of 20%
    return max(0.2, expected_loss_ratio)


def sample_actual_loss_ratio(strength_ratio, variance_factor=5.0):
    """
    Sample an actual loss ratio from a distribution centered on the expected loss ratio.
    Used when a battle is actually fought to determine real casualties.

    Args:
        strength_ratio: Ratio of attacker strength to defender strength
        variance_factor: Controls how tight the distribution is around the expected value

    Returns:
        float: Actual proportion of attacking units lost in this specific battle (0.2-0.8)
    """
    # Calculate the expected loss ratio as the mean of our distribution
    expected_loss = calculate_expected_loss_ratio(strength_ratio)

    # As strength_ratio increases, variance should decrease
    variance = 0.3 / math.sqrt(strength_ratio)

    # Use a Beta distribution which is naturally bounded between 0 and 1
    # We'll scale and shift it to fit our 0.2-0.8 range

    # Calculate alpha and beta parameters for a Beta distribution centered on our expected loss
    mean = (expected_loss - 0.2) / 0.6  # Scale expected_loss to 0-1 range

    # Higher variance_factor = tighter distribution
    alpha = mean * variance_factor
    beta = (1 - mean) * variance_factor

    # Sample from Beta distribution
    sampled_value = random.betavariate(max(0.1, alpha), max(0.1, beta))

    # Convert back to our 0.2-0.8 range
    actual_loss = 0.2 + sampled_value * 0.6

    return actual_loss


def should_attack(
    observation: Observation, orig_pos: tuple[int, int], dest_pos: tuple[int, int], unit_type_idx: int, threshold: float = 0.6
) -> bool:
    """
    Determines if an attack from orig_pos to dest_pos using the specified unit type
    would be advantageous.

    Args:
        observation: The current observation
        orig_pos: (row, col) of the origin position
        dest_pos: (row, col) of the destination position
        unit_type_idx: Index of the unit type to use (0: cavalry, 1: infantry, etc.)
        threshold: Minimum win probability required to recommend attack

    Returns:
        bool: True if attack is recommended, False otherwise
    """
    orig_row, orig_col = orig_pos

    # Get the unit array for this unit type
    unit_arrays = [observation.cavalry, observation.infantry, observation.archers, observation.siege]
    unit_array = unit_arrays[unit_type_idx]
    unit_type = UNIT_TYPES[unit_type_idx]

    # Get attacking unit count (leave at least 1 unit behind)
    attacking_unit_count = unit_array[orig_row, orig_col] - 1

    # Don't attack if we don't have enough units
    if attacking_unit_count <= 0:
        return False

    # Predict combat outcome
    win_probability, _ = predict_combat_outcome(observation, orig_pos, dest_pos, unit_type, attacking_unit_count)

    # Return True if win probability exceeds threshold
    return win_probability >= threshold
