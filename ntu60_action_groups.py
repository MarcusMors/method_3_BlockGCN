"""
NTU60 Action Class Definitions and Group Mappings
=================================================
Standard NTU RGB+D 60-action labels with their category groupings.

The 60 actions are divided into 3 major categories (as per the original paper):
  - Daily Actions (0-26):    27 classes — everyday single-person activities
  - Health-related (27-40):  14 classes — physical reactions / health-adjacent actions
  - Mutual Actions (41-59):  19 classes — actions involving interaction with another person
"""

# Complete NTU60 action list (0-indexed)
NTU60_ACTIONS = {
    0: "drink water",
    1: "eat meal/snack",
    2: "brushing teeth",
    3: "brushing hair",
    4: "drop",
    5: "pickup",
    6: "throw",
    7: "sitting down",
    8: "standing up (from sitting)",
    9: "clapping",
    10: "reading",
    11: "writing",
    12: "tear up paper",
    13: "wear jacket",
    14: "take off jacket",
    15: "wear a shoe",
    16: "take off a shoe",
    17: "wear on glasses",
    18: "take off glasses",
    19: "put on a hat/cap",
    20: "take off a hat/cap",
    21: "cheer up",
    22: "hand waving",
    23: "kicking something",
    24: "reach into pocket",
    25: "hopping (one foot)",
    26: "jump up",
    27: "make a phone call",
    28: "playing with phone",
    29: "type on a keyboard",
    30: "point to something",
    31: "taking a selfie",
    32: "check time (from watch)",
    33: "rub two hands together",
    34: "nod head/bow",
    35: "shake head",
    36: "wipe face",
    37: "salute",
    38: "put the palms together",
    39: "cross hands in front",
    40: "sneeze/cough",
    41: "staggering",
    42: "falling down",
    43: "touch head (headache)",
    44: "touch chest (pain)",
    45: "touch back (backache)",
    46: "touch neck (neckache)",
    47: "nausea/vomiting",
    48: "use a fan",
    49: "punching/slapping",
    50: "kicking other person",
    51: "pushing other person",
    52: "pat on back of other",
    53: "point finger at other",
    54: "hugging other person",
    55: "giving something",
    56: "touch other's pocket",
    57: "handshaking",
    58: "walking towards other",
    59: "walking apart from other",
}

# Action group definitions (as per NTU RGB+D paper)
ACTION_GROUPS = {
    "Daily": list(range(0, 27)),      # 0-26  (27 classes)
    "Health": list(range(27, 41)),    # 27-40 (14 classes)
    "Mutual": list(range(41, 60)),    # 41-59 (19 classes)
}

# Short names for plots
ACTION_SHORT_NAMES = {
    0: "drink",
    1: "eat",
    2: "brush_teeth",
    3: "brush_hair",
    4: "drop",
    5: "pickup",
    6: "throw",
    7: "sit_down",
    8: "stand_up",
    9: "clap",
    10: "read",
    11: "write",
    12: "tear_paper",
    13: "wear_jacket",
    14: "off_jacket",
    15: "wear_shoe",
    16: "off_shoe",
    17: "wear_glasses",
    18: "off_glasses",
    19: "wear_hat",
    20: "off_hat",
    21: "cheer",
    22: "wave",
    23: "kick_obj",
    24: "reach_pocket",
    25: "hop",
    26: "jump",
    27: "phone_call",
    28: "play_phone",
    29: "type",
    30: "point",
    31: "selfie",
    32: "check_time",
    33: "rub_hands",
    34: "nod",
    35: "shake_head",
    36: "wipe_face",
    37: "salute",
    38: "palms_together",
    39: "cross_hands",
    40: "sneeze",
    41: "stagger",
    42: "fall",
    43: "touch_head",
    44: "touch_chest",
    45: "touch_back",
    46: "touch_neck",
    47: "nausea",
    48: "use_fan",
    49: "punch",
    50: "kick_person",
    51: "push",
    52: "pat_back",
    53: "point_finger",
    54: "hug",
    55: "give_obj",
    56: "touch_pocket",
    57: "handshake",
    58: "walk_toward",
    59: "walk_apart",
}


def get_action_name(class_idx):
    """Return the full action name for a given class index."""
    return NTU60_ACTIONS.get(class_idx, f"Unknown-{class_idx}")


def get_short_name(class_idx):
    """Return a short action name suitable for plots."""
    return ACTION_SHORT_NAMES.get(class_idx, f"cls{class_idx}")


def get_group_name(class_idx):
    """Return the group name ('Daily', 'Health', 'Mutual') for a class index."""
    for group_name, indices in ACTION_GROUPS.items():
        if class_idx in indices:
            return group_name
    return "Unknown"


def get_group_indices(group_name):
    """Return list of class indices belonging to a group."""
    return ACTION_GROUPS.get(group_name, [])


if __name__ == "__main__":
    # Sanity check
    print("NTU60 Action Groups Summary")
    print("=" * 50)
    for group, indices in ACTION_GROUPS.items():
        print(f"\n{group} Actions ({len(indices)} classes):")
        for idx in indices:
            print(f"  {idx:2d}: {NTU60_ACTIONS[idx]}")
    print(f"\nTotal: {sum(len(v) for v in ACTION_GROUPS.values())} classes")
