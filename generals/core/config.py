from enum import Enum, IntEnum, StrEnum
from importlib.resources import files
from typing import Literal

# Game Literals
PASSABLE: Literal["."] = "."
MOUNTAIN: Literal["#"] = "#"


class UnitType(Enum):
    CAVALRY = 0
    INFANTRY = 1
    ARCHERS = 2
    SIEGE = 3


class Dimension(IntEnum):
    SQUARE_SIZE = 50
    GUI_CELL_HEIGHT = 30
    GUI_CELL_WIDTH = 70
    MINIMUM_WINDOW_SIZE = 700


class Direction(Enum):
    UP = (-1, 0)
    DOWN = (1, 0)
    LEFT = (0, -1)
    RIGHT = (0, 1)


DIRECTIONS = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]


class Path(StrEnum):
    GENERAL_PATH = str(files("generals.assets.images") / "crownie.png")
    CITY_PATH = str(files("generals.assets.images") / "citie.png")
    MOUNTAIN_PATH = str(files("generals.assets.images") / "mountainie.png")

    # Unit type icons (these would need to be created)
    CAVALRY_PATH = str(files("generals.assets.images") / "cavalry.png")
    INFANTRY_PATH = str(files("generals.assets.images") / "infantry.png")
    ARCHERS_PATH = str(files("generals.assets.images") / "archers.png")
    SIEGE_PATH = str(files("generals.assets.images") / "siege.png")

    # Font options are Quicksand-SemiBold.ttf, Quicksand-Medium.ttf, Quicksand-Light.ttf
    FONT_PATH = str(files("generals.assets.fonts") / "Quicksand-Medium.ttf")
