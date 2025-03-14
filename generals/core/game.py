from typing import Any, TypeAlias

import numba as nb
import numpy as np

from .action import Action
from .channels import Channels, UNIT_TYPES, COMBAT_EFFECTIVENESS
from .config import DIRECTIONS
from .grid import Grid
from .observation import Observation

# Type aliases
Info: TypeAlias = dict[str, Any]


@nb.njit(cache=True)
def calculate_army_size(armies, ownership):
    return np.int32(np.sum(armies * ownership))


@nb.njit(cache=True)
def calculate_land_size(ownership):
    return np.int32(np.sum(ownership))


class Game:
    def __init__(self, grid: Grid, agents: list[str]):
        # Agents
        self.agents = agents
        self.agent_order = self.agents[:]

        # Grid
        _grid = grid.grid
        self.channels = Channels(_grid, self.agents)
        self.grid_dims = (_grid.shape[0], _grid.shape[1])
        self.general_positions = {agent: np.argwhere(_grid == chr(ord("A") + i))[0] for i, agent in enumerate(self.agents)}

        # Time stuff
        self.time = 0
        self.increment_rate = 50

        # Limits
        self.max_army_value = 100_000
        self.max_land_value = np.prod(self.grid_dims)
        self.max_timestep = 100_000

        self.winner = None
        self.loser = None

    def resolve_combat(
        self,
        attacking_agent: str,
        defending_agent: str,
        attacker_pos: tuple[int, int],
        defender_pos: tuple[int, int],
    ) -> tuple[str, dict[str, float]]:
        """
        Resolve combat between attacking and defending armies using unit type effectiveness.

        Args:
            attacking_agent: ID of the attacking agent
            defending_agent: ID of the defending agent
            attacker_pos: (row, col) of attacking position
            defender_pos: (row, col) of defending position

        Returns:
            tuple: (winner_agent, remaining_units)
                winner_agent: ID of the winning agent
                remaining_units: Dictionary of remaining unit counts by type
        """
        # Get the counts of each unit type
        attacker_units = {}
        defender_units = {}

        # Get unit counts for attacker
        attacker_units["cavalry"] = self.channels.cavalry[attacker_pos]
        attacker_units["infantry"] = self.channels.infantry[attacker_pos]
        attacker_units["archers"] = self.channels.archers[attacker_pos]
        attacker_units["siege"] = self.channels.siege[attacker_pos]

        # Get unit counts for defender
        defender_units["cavalry"] = self.channels.cavalry[defender_pos]
        defender_units["infantry"] = self.channels.infantry[defender_pos]
        defender_units["archers"] = self.channels.archers[defender_pos]
        defender_units["siege"] = self.channels.siege[defender_pos]

        # Calculate combat power
        attacker_power = 0
        for att_type, att_count in attacker_units.items():
            unit_contribution = 0
            for def_type, def_count in defender_units.items():
                effectiveness = COMBAT_EFFECTIVENESS[att_type][def_type]
                unit_contribution += effectiveness * def_count
            attacker_power += att_count * unit_contribution

        defender_power = 0
        for def_type, def_count in defender_units.items():
            unit_contribution = 0
            for att_type, att_count in attacker_units.items():
                effectiveness = COMBAT_EFFECTIVENESS[def_type][att_type]
                unit_contribution += effectiveness * att_count
            defender_power += def_count * unit_contribution

        # Calculate combat result
        attacker_total = sum(attacker_units.values())
        defender_total = sum(defender_units.values())

        # Avoid division by zero
        if attacker_total == 0:
            return defending_agent, defender_units
        if defender_total == 0:
            return attacking_agent, attacker_units

        # Calculate remaining percentage for winner
        remaining_percentage = 0
        winner_agent = ""

        if attacker_power > defender_power:
            winner_agent = attacking_agent
            # Attacker wins - calculate remaining percentage
            loss_ratio = defender_power / attacker_power
            remaining_percentage = 1 - (loss_ratio * 0.8)  # 80% effectiveness to avoid complete wipeouts
        else:
            winner_agent = defending_agent
            # Defender wins - calculate remaining percentage
            loss_ratio = attacker_power / defender_power
            remaining_percentage = 1 - (loss_ratio * 0.5)  # Defenders lose fewer units (50% effectiveness)

        # Ensure some minimal survival rate
        remaining_percentage = max(0.1, remaining_percentage)

        # Calculate remaining units
        remaining_units = {}

        if winner_agent == attacking_agent:
            # Attacker wins - convert units based on remaining percentage
            for unit_type, unit_count in attacker_units.items():
                remaining_units[unit_type] = unit_count * remaining_percentage
        else:
            # Defender wins - keep units based on remaining percentage
            for unit_type, unit_count in defender_units.items():
                remaining_units[unit_type] = unit_count * remaining_percentage

        return winner_agent, remaining_units

    def is_done(self) -> bool:
        return self.winner is not None

    def get_infos(self) -> dict[str, Info]:
        """
        Returns a dictionary of player statistics.
        Keys and values are as follows:
        - army: total army size
        - land: total land size
        - is_done: True if the game is over, False otherwise
        - is_winner: True if the player won, False otherwise
        """
        players_stats = {}
        for agent in self.agents:
            army_size = calculate_army_size(self.channels.armies, self.channels.ownership[agent])
            land_size = calculate_land_size(self.channels.ownership[agent])
            players_stats[agent] = {
                "army": army_size,
                "land": land_size,
                "is_done": self.is_done(),
                "is_winner": self.winner == agent,
            }
        return players_stats

    def step(self, actions: dict[str, Action]) -> tuple[dict[str, Observation], dict[str, Any]]:
        """
        Perform one step of the game
        """
        done_before_actions = self.is_done()
        for agent in self.agent_order:
            pass_turn, si, sj, direction, unit_type_idx, split_army = actions[agent]

            # Get the unit type being moved
            unit_type = UNIT_TYPES[unit_type_idx]

            # Get the current unit array based on unit type
            if unit_type == "cavalry":
                unit_array = self.channels.cavalry
            elif unit_type == "infantry":
                unit_array = self.channels.infantry
            elif unit_type == "archers":
                unit_array = self.channels.archers
            elif unit_type == "siege":
                unit_array = self.channels.siege
            else:
                continue  # Invalid unit type

            # Skip if agent wants to pass the turn
            if pass_turn == 1:
                continue

            if split_army == 1:  # Agent wants to split the army
                army_to_move = unit_array[si, sj] / 2.0
            else:  # Leave just one army in the source cell
                army_to_move = unit_array[si, sj] - 1

            if army_to_move < 1.0:  # Skip if army size to move is less than 1
                continue

            # Cap the amount of army to move (previous moves may have lowered available army)
            army_to_move = min(army_to_move, unit_array[si, sj] - 1)
            army_to_stay = unit_array[si, sj] - army_to_move

            # Check if the current agent still owns the source cell and has more than 1 army
            if self.channels.ownership[agent][si, sj] == 0 or army_to_move < 1:
                continue

            di, dj = (
                si + DIRECTIONS[direction].value[0],
                sj + DIRECTIONS[direction].value[1],
            )  # destination indices

            # Skip if the destination cell is not passable or out of bounds
            if di < 0 or di >= self.grid_dims[0] or dj < 0 or dj >= self.grid_dims[1]:
                continue
            if self.channels.passable[di, dj] == 0:
                continue

            # Figure out the target square owner
            target_square_owner_idx = np.argmax(
                [self.channels.ownership[agent][di, dj] for agent in ["neutral"] + self.agents]
            )
            target_square_owner = (["neutral"] + self.agents)[target_square_owner_idx]

            # Update source cell - remove moving units
            unit_array[si, sj] = army_to_stay

            if target_square_owner == agent:
                # Moving to own cell - just add units
                unit_array[di, dj] += army_to_move
            else:
                # Moving to enemy/neutral cell - resolve combat
                source_pos = (si, sj)
                target_pos = (di, dj)

                # Save counts before combat
                attacker_units = {unit_type: army_to_move}

                # Temporarily place attacking units at target to resolve combat
                original_target_cavalry = self.channels.cavalry[di, dj]
                original_target_infantry = self.channels.infantry[di, dj]
                original_target_archers = self.channels.archers[di, dj]
                original_target_siege = self.channels.siege[di, dj]

                # Set attacking units at target temporarily
                self.channels.cavalry[di, dj] = army_to_move if unit_type == "cavalry" else 0
                self.channels.infantry[di, dj] = army_to_move if unit_type == "infantry" else 0
                self.channels.archers[di, dj] = army_to_move if unit_type == "archers" else 0
                self.channels.siege[di, dj] = army_to_move if unit_type == "siege" else 0

                # Resolve combat
                square_winner, remaining_units = self.resolve_combat(agent, target_square_owner, source_pos, target_pos)

                # Restore target cell units
                self.channels.cavalry[di, dj] = original_target_cavalry
                self.channels.infantry[di, dj] = original_target_infantry
                self.channels.archers[di, dj] = original_target_archers
                self.channels.siege[di, dj] = original_target_siege

                # Update target cell based on combat result
                if square_winner == agent:
                    # Attacker won - update cell ownership and set remaining units
                    self.channels.ownership[agent][di, dj] = True
                    if target_square_owner != "neutral":
                        self.channels.ownership[target_square_owner][di, dj] = False

                    # Set the remaining units
                    self.channels.cavalry[di, dj] = remaining_units.get("cavalry", 0)
                    self.channels.infantry[di, dj] = remaining_units.get("infantry", 0)
                    self.channels.archers[di, dj] = remaining_units.get("archers", 0)
                    self.channels.siege[di, dj] = remaining_units.get("siege", 0)

                    # Check if the captured cell is the opponent's general
                    if target_square_owner in self.agents:
                        gi, gj = tuple(self.general_positions[target_square_owner])
                        if (di, dj) == (gi, gj):
                            self.loser = target_square_owner
                            self.winner = agent
                else:
                    # Defender won - update with remaining units
                    self.channels.cavalry[di, dj] = remaining_units.get("cavalry", 0)
                    self.channels.infantry[di, dj] = remaining_units.get("infantry", 0)
                    self.channels.archers[di, dj] = remaining_units.get("archers", 0)
                    self.channels.siege[di, dj] = remaining_units.get("siege", 0)

        # Swap agent order (because priority is alternating)
        self.agent_order = self.agent_order[::-1]

        if not done_before_actions:
            self.time += 1

        if self.is_done():
            # give all cells of loser to winner
            winner = self.agents[0] if self.winner == self.agents[0] else self.agents[1]
            loser = self.agents[1] if winner == self.agents[0] else self.agents[0]
            self.channels.ownership[winner] += self.channels.ownership[loser]
            self.channels.ownership[loser] = np.full(self.grid_dims, False)
        else:
            self._global_game_update()

        observations = {agent: self.agent_observation(agent) for agent in self.agents}
        infos = self.get_infos()
        return observations, infos

    def _global_game_update(self) -> None:
        """
        Update game state globally.
        """
        owners = self.agents

        # every `increment_rate` steps, increase army size in each cell
        if self.time % self.increment_rate == 0:
            for owner in owners:
                for unit_array in [self.channels.cavalry, self.channels.infantry, self.channels.archers, self.channels.siege]:
                    unit_array += self.channels.ownership[owner]

        # Increment armies on general and city cells, but only if they are owned by player
        if self.time % 2 == 0 and self.time > 0:
            update_mask = self.channels.generals + self.channels.cities
            for owner in owners:
                # Generals produce infantry, cities produce a mix of units
                self.channels.infantry += update_mask * self.channels.ownership[owner]

                # Cities also produce some cavalry and archers (less than infantry)
                city_mask = self.channels.cities * self.channels.ownership[owner]
                if self.time % 6 == 0:  # Every 6 turns, cities produce cavalry
                    self.channels.cavalry += city_mask
                if self.time % 8 == 0:  # Every 8 turns, cities produce archers
                    self.channels.archers += city_mask

    def agent_observation(self, agent: str) -> Observation:
        """
        Returns an observation for a given agent.
        """
        scores = {}
        for _agent in self.agents:
            # Calculate total army size across all unit types
            army_size = (
                np.sum(self.channels.cavalry * self.channels.ownership[_agent])
                + np.sum(self.channels.infantry * self.channels.ownership[_agent])
                + np.sum(self.channels.archers * self.channels.ownership[_agent])
                + np.sum(self.channels.siege * self.channels.ownership[_agent])
            ).astype(int)

            land_size = np.sum(self.channels.ownership[_agent]).astype(int)
            scores[_agent] = {
                "army": army_size,
                "land": land_size,
            }

        visible = self.channels.get_visibility(agent)
        invisible = 1 - visible

        opponent = self.agents[0] if agent == self.agents[1] else self.agents[1]

        # Unit arrays
        cavalry = self.channels.cavalry * visible
        infantry = self.channels.infantry * visible
        archers = self.channels.archers * visible
        siege = self.channels.siege * visible

        # Total armies (sum of all unit types)
        armies = cavalry + infantry + archers + siege

        # Original observation fields
        mountains = self.channels.mountains * visible
        generals = self.channels.generals * visible
        cities = self.channels.cities * visible
        neutral_cells = self.channels.ownership_neutral * visible
        owned_cells = self.channels.ownership[agent] * visible
        opponent_cells = self.channels.ownership[opponent] * visible
        structures_in_fog = invisible * (self.channels.mountains + self.channels.cities)
        fog_cells = invisible - structures_in_fog
        owned_land_count = scores[agent]["land"]
        owned_army_count = scores[agent]["army"]
        opponent_land_count = scores[opponent]["land"]
        opponent_army_count = scores[opponent]["army"]
        timestep = self.time
        priority = 1 if agent == self.agent_order[0] else 0

        return Observation(
            # Unit types
            cavalry=cavalry,
            infantry=infantry,
            archers=archers,
            siege=siege,
            # Total armies (for backward compatibility)
            armies=armies,
            # Original observation fields
            generals=generals,
            cities=cities,
            mountains=mountains,
            neutral_cells=neutral_cells,
            owned_cells=owned_cells,
            opponent_cells=opponent_cells,
            fog_cells=fog_cells,
            structures_in_fog=structures_in_fog,
            owned_land_count=owned_land_count,
            owned_army_count=owned_army_count,
            opponent_land_count=opponent_land_count,
            opponent_army_count=opponent_army_count,
            timestep=timestep,
            priority=priority,
        )
