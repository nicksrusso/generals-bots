import numpy as np
from scipy.ndimage import maximum_filter  # type: ignore

from .config import MOUNTAIN, PASSABLE

valid_generals = ["A", "B"]  # Generals are represented by A and B

# Define unit types
UNIT_TYPES = ["cavalry", "infantry", "archers", "siege"]

# Combat effectiveness matrix (attacking unit type vs defending unit type)
COMBAT_EFFECTIVENESS = {
    "cavalry": {"cavalry": 1.0, "infantry": 0.75, "archers": 1.25, "siege": 1.5},
    "infantry": {"cavalry": 1.25, "infantry": 1.0, "archers": 0.75, "siege": 1.0},
    "archers": {"cavalry": 1.5, "infantry": 0.75, "archers": 1.0, "siege": 1.25},
    "siege": {"cavalry": 0.5, "infantry": 0.75, "archers": 0.75, "siege": 1.0},
}


class Channels:
    """
    Unit arrays - separate array for each unit type:
      - cavalry: fast unit, strong vs archers, weak vs infantry
      - infantry: balanced unit, strong vs cavalry, weak vs archers
      - archers: ranged unit, strong vs cavalry, weak vs infantry
      - siege: special unit, strong vs structures, weak in direct combat

    generals - general mask (1 if general is in cell, 0 otherwise)
    mountains - mountain mask (1 if mountain is in cell, 0 otherwise)
    cities - city mask (1 if city is in cell, 0 otherwise)
    passable - passable mask (1 if cell is passable, 0 otherwise)
    ownership_i - ownership mask for player i (1 if player i owns cell, 0 otherwise)
    ownership_neutral - ownership mask for neutral cells that are
    passable (1 if cell is neutral, 0 otherwise)
    """

    def __init__(self, grid: np.ndarray, _agents: list[str]):
        # Initialize unit arrays - instead of one armies array, create separate arrays for each unit type
        self._cavalry = np.zeros_like(grid, dtype=np.float16)
        self._infantry = np.zeros_like(grid, dtype=np.float16)
        self._archers = np.zeros_like(grid, dtype=np.float16)
        self._siege = np.zeros_like(grid, dtype=np.float16)

        # Initialize with one default unit type at general positions (infantry)
        self._infantry = np.where(np.isin(grid, valid_generals), 1, 0).astype(np.float16)

        self._generals = np.where(np.isin(grid, valid_generals), 1, 0).astype(bool)
        self._mountains = np.where(grid == MOUNTAIN, 1, 0).astype(bool)
        self._passable = (grid != MOUNTAIN).astype(bool)
        self._cities = np.where(np.char.isdigit(grid), 1, 0).astype(bool)
        self._cities = (self._cities + np.where(grid == "x", 1, 0)).astype(bool)  # city with value 50 is marked as x

        self._ownership = {"neutral": ((grid == PASSABLE) | (np.char.isdigit(grid)) | (grid == "x")).astype(bool)}
        for i, agent in enumerate(_agents):
            self._ownership[agent] = np.where(grid == chr(ord("A") + i), 1, 0).astype(bool)

        # City costs are 40 + digit in the cell - add city populations as infantry
        city_costs = np.where(np.char.isdigit(grid), grid, "0").astype(np.float16)
        city_costs += np.where(grid == "x", 10, 0)
        # Add city units as infantry
        city_population = 40 * self._cities + city_costs
        self._infantry += city_population

    def get_total_armies(self) -> np.ndarray:
        """Returns the total number of units in each cell (all unit types combined)"""
        return self._cavalry + self._infantry + self._archers + self._siege

    def get_unit_counts(self, position: tuple[int, int]) -> dict[str, float]:
        """Returns counts of each unit type at the specified position"""
        i, j = position
        return {
            "cavalry": self._cavalry[i, j],
            "infantry": self._infantry[i, j],
            "archers": self._archers[i, j],
            "siege": self._siege[i, j],
        }

    def calculate_combat_power(self, position: tuple[int, int], enemy_position: tuple[int, int]) -> float:
        """Calculate the combat power of units at position against units at enemy_position"""
        friendly_units = self.get_unit_counts(position)
        enemy_units = self.get_unit_counts(enemy_position)

        total_power = 0
        for friendly_type, friendly_count in friendly_units.items():
            unit_contribution = 0
            for enemy_type, enemy_count in enemy_units.items():
                effectiveness = COMBAT_EFFECTIVENESS[friendly_type][enemy_type]
                unit_contribution += effectiveness * enemy_count
            total_power += friendly_count * unit_contribution

        return total_power

    def get_visibility(self, agent_id: str) -> np.ndarray:
        channel = self._ownership[agent_id]
        return maximum_filter(channel, size=3).astype(bool)

    @staticmethod
    def channel_to_indices(channel: np.ndarray) -> np.ndarray:
        """
        Returns a list of indices of cells with non-zero values from specified a channel.
        """
        return np.argwhere(channel != 0)

    # Property getters and setters for unit types
    @property
    def cavalry(self) -> np.ndarray:
        return self._cavalry

    @cavalry.setter
    def cavalry(self, value):
        self._cavalry = value

    @property
    def infantry(self) -> np.ndarray:
        return self._infantry

    @infantry.setter
    def infantry(self, value):
        self._infantry = value

    @property
    def archers(self) -> np.ndarray:
        return self._archers

    @archers.setter
    def archers(self, value):
        self._archers = value

    @property
    def siege(self) -> np.ndarray:
        return self._siege

    @siege.setter
    def siege(self, value):
        self._siege = value

    @property
    def armies(self) -> np.ndarray:
        """For backward compatibility - returns the total armies"""
        return self.get_total_armies()

    @property
    def ownership(self) -> dict[str, np.ndarray]:
        return self._ownership

    @ownership.setter
    def ownership(self, value):
        self._ownership = value

    @property
    def generals(self) -> np.ndarray:
        return self._generals

    @generals.setter
    def generals(self, value):
        self._generals = value

    @property
    def mountains(self) -> np.ndarray:
        return self._mountains

    @mountains.setter
    def mountains(self, value):
        self._mountains = value

    @property
    def cities(self) -> np.ndarray:
        return self._cities

    @cities.setter
    def cities(self, value):
        self._cities = value

    @property
    def passable(self) -> np.ndarray:
        return self._passable

    @passable.setter
    def passable(self, value):
        self._passable = value

    @property
    def ownership_neutral(self) -> np.ndarray:
        return self._ownership["neutral"]

    @ownership_neutral.setter
    def ownership_neutral(self, value):
        self._ownership["neutral"] = value
