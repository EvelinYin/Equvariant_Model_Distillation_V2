from src.equ_lib.groups.flipping_group import FlipGroup
from src.equ_lib.groups.rot90_group import Rot90Group
from src.equ_lib.groups.rot45_group import Rot45Group

__GROUPS__ = {
    "FlipGroup": FlipGroup,
    "Rot90Group": Rot90Group,
    "Rot45Group": Rot45Group,
}


def get_group(name: str):
    if name not in __GROUPS__:
        raise ValueError(f"Group '{name}' not found. Available: {list(__GROUPS__.keys())}")
    return __GROUPS__[name]()