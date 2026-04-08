DEFAULT_LABEL_MODE = "binary"


LABEL_MODE_SPECS = {
    "binary": {
        "label_map": {
            1: 0,
            2: 1,
            3: 1,
            4: 1,
            5: 1,
            6: 1,
        },
        "class_names": ["rest", "active"],
    },
    "rest_vs_active": {
        "label_map": {
            1: 0,
            2: 1,
            3: 1,
            4: 1,
            5: 1,
            6: 1,
        },
        "class_names": ["rest", "active"],
    },
    "rest_vs_active": {
        "label_map": {
            1: 0,
            2: 1,
            3: 1,
            4: 1,
            5: 1,
            6: 1,
        },
        "class_names": ["rest", "active"],
    },
    "rest_2_3": {
        "label_map": {
            1: 0,
            2: 1,
            3: 2,
        },
        "class_names": ["rest", "fist", "wrist_flexion"],
    },
    "full_6": {
        "label_map": {
            1: 0,
            2: 1,
            3: 2,
            4: 3,
            5: 4,
            6: 5,
        },
        "class_names": [
            "rest",
            "fist",
            "wrist_flexion",
            "wrist_extension",
            "radial_deviation",
            "ulnar_deviation",
        ],
    },
    "full_7_optional": {
        "label_map": {
            1: 0,
            2: 1,
            3: 2,
            4: 3,
            5: 4,
            6: 5,
            7: 6,
        },
        "class_names": [
            "rest",
            "fist",
            "wrist_flexion",
            "wrist_extension",
            "radial_deviation",
            "ulnar_deviation",
            "extended_palm",
        ],
    },
}


def get_label_mode_spec(label_mode):
    if label_mode not in LABEL_MODE_SPECS:
        valid = ", ".join(sorted(LABEL_MODE_SPECS))
        raise ValueError(f"Unknown label mode '{label_mode}'. Valid options: {valid}.")
    return LABEL_MODE_SPECS[label_mode]


def get_label_map(label_mode):
    return get_label_mode_spec(label_mode)["label_map"]


def get_class_names(label_mode):
    return get_label_mode_spec(label_mode)["class_names"]


def num_classes_for_mode(label_mode):
    return len(get_class_names(label_mode))
