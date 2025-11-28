#!/usr/bin/env python3
"""
scenarios.py - Scenario definitions and experiment generators for keystroke biometrics.

This module defines the 6 experimental scenarios and generates all sub-experiments
for each scenario. Results are aggregated at the scenario level.

Scenarios:
    1: Same platform, same topic (train S1 <-> test S2, bidirectional)
    2: Same platform, cross topic (train 2 videos -> test 1 video)
    3.1: Cross platform, same topic (1-1) - one platform to one platform
    3.2: Cross platform, same topic (2-1) - two platforms to one platform
    4.1: Cross platform, cross topic (1-1) - different platform AND video
    4.2: Cross platform, cross topic (2-1) - two platforms to one, different video

Notation:
    P = Platform (F=1=Facebook, I=2=Instagram, T=3=Twitter)
    V = Video/Topic (1, 2, 3)
    S = Session (1, 2)
"""

from itertools import combinations, permutations
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, field


# Constants for readability
PLATFORMS = [1, 2, 3]  # F=1, I=2, T=3
VIDEOS = [1, 2, 3]
SESSIONS = [1, 2]
ALL_SESSIONS = [1, 2]

PLATFORM_NAMES = {1: "F", 2: "I", 3: "T"}
VIDEO_NAMES = {1: "V1", 2: "V2", 3: "V3"}
SESSION_NAMES = {1: "S1", 2: "S2"}


@dataclass
class SubExperiment:
    """Represents a single sub-experiment within a scenario."""
    name: str
    scenario_id: str
    train_filter: Dict[str, List[int]]  # e.g., {"platform_id": [1], "video_id": [1, 2], "session_id": [1, 2]}
    test_filter: Dict[str, List[int]]   # e.g., {"platform_id": [2], "video_id": [3], "session_id": [1, 2]}

    def get_train_mask_conditions(self) -> List[Tuple[str, List[int]]]:
        """Get list of (column, values) for train filtering."""
        return [(col, vals) for col, vals in self.train_filter.items()]

    def get_test_mask_conditions(self) -> List[Tuple[str, List[int]]]:
        """Get list of (column, values) for test filtering."""
        return [(col, vals) for col, vals in self.test_filter.items()]


@dataclass
class Scenario:
    """Represents a complete scenario with all its sub-experiments."""
    id: str
    name: str
    description: str
    sub_experiments: List[SubExperiment] = field(default_factory=list)

    def __len__(self):
        return len(self.sub_experiments)


def _format_filter_name(filter_dict: Dict[str, List[int]]) -> str:
    """Format a filter dictionary into a readable name."""
    parts = []
    if "platform_id" in filter_dict:
        p_names = "".join(PLATFORM_NAMES[p] for p in filter_dict["platform_id"])
        parts.append(f"P{p_names}")
    if "video_id" in filter_dict:
        v_names = "".join(str(v) for v in filter_dict["video_id"])
        parts.append(f"V{v_names}")
    if "session_id" in filter_dict:
        s_names = "".join(str(s) for s in filter_dict["session_id"])
        parts.append(f"S{s_names}")
    return "_".join(parts)


def generate_scenario_1() -> Scenario:
    """
    Scenario 1: Same platform, same topic

    Train on session 1, test on session 2 (and vice versa).
    For each combination of platform and video.

    Sub-experiments: 3 platforms × 3 videos × 2 directions = 18
    """
    scenario = Scenario(
        id="1",
        name="Scenario 1",
        description="Same platform, same topic"
    )

    for platform in PLATFORMS:
        for video in VIDEOS:
            # Direction 1: Session1 -> Session2
            sub_exp_1 = SubExperiment(
                name=f"Scenario1_{PLATFORM_NAMES[platform]}_Video{video}_Session1 vs Session2",
                scenario_id="1",
                train_filter={
                    "platform_id": [platform],
                    "video_id": [video],
                    "session_id": [1]
                },
                test_filter={
                    "platform_id": [platform],
                    "video_id": [video],
                    "session_id": [2]
                }
            )
            scenario.sub_experiments.append(sub_exp_1)

            # Direction 2: Session2 -> Session1 (reverse)
            sub_exp_2 = SubExperiment(
                name=f"Scenario1_{PLATFORM_NAMES[platform]}_Video{video}_Session2 vs Session1",
                scenario_id="1",
                train_filter={
                    "platform_id": [platform],
                    "video_id": [video],
                    "session_id": [2]
                },
                test_filter={
                    "platform_id": [platform],
                    "video_id": [video],
                    "session_id": [1]
                }
            )
            scenario.sub_experiments.append(sub_exp_2)

    return scenario


def generate_scenario_2() -> Scenario:
    """
    Scenario 2: Same platform, cross topic

    Train on 2 videos (all sessions), test on 1 video (all sessions).
    Leave-one-video-out within each platform.

    Sub-experiments: 3 platforms × 3 leave-one-out combinations = 9
    """
    scenario = Scenario(
        id="2",
        name="Scenario 2",
        description="Same platform, cross topic"
    )

    for platform in PLATFORMS:
        for test_video in VIDEOS:
            train_videos = [v for v in VIDEOS if v != test_video]

            train_video_names = " and ".join(f"Video{v}" for v in train_videos)
            sub_exp = SubExperiment(
                name=f"Scenario2_{PLATFORM_NAMES[platform]}_{train_video_names} vs {PLATFORM_NAMES[platform]}_Video{test_video}",
                scenario_id="2",
                train_filter={
                    "platform_id": [platform],
                    "video_id": train_videos,
                    "session_id": ALL_SESSIONS
                },
                test_filter={
                    "platform_id": [platform],
                    "video_id": [test_video],
                    "session_id": ALL_SESSIONS
                }
            )
            scenario.sub_experiments.append(sub_exp)

    return scenario


def generate_scenario_3_1() -> Scenario:
    """
    Scenario 3.1: Cross platform, same topic (1-1)

    Train on 1 platform, test on 1 different platform.
    Same video, all sessions.

    Sub-experiments: 3 platform pairs (F-I, F-T, I-T) × 3 videos = 9
    """
    scenario = Scenario(
        id="3.1",
        name="Scenario 3.1",
        description="Cross platform, same topic (1-1)"
    )

    # Platform pairs (ordered: train -> test)
    platform_pairs = [
        (1, 2),  # F -> I
        (1, 3),  # F -> T
        (2, 3),  # I -> T
    ]

    for train_p, test_p in platform_pairs:
        for video in VIDEOS:
            sub_exp = SubExperiment(
                name=f"Scenario3.1_{PLATFORM_NAMES[train_p]} vs {PLATFORM_NAMES[test_p]}_Video{video}",
                scenario_id="3.1",
                train_filter={
                    "platform_id": [train_p],
                    "video_id": [video],
                    "session_id": ALL_SESSIONS
                },
                test_filter={
                    "platform_id": [test_p],
                    "video_id": [video],
                    "session_id": ALL_SESSIONS
                }
            )
            scenario.sub_experiments.append(sub_exp)

    return scenario


def generate_scenario_3_2() -> Scenario:
    """
    Scenario 3.2: Cross platform, same topic (2-1)

    Train on 2 platforms, test on 1 remaining platform.
    Same video, all sessions. Leave-one-platform-out.

    Sub-experiments: 3 leave-one-out × 3 videos = 9
    """
    scenario = Scenario(
        id="3.2",
        name="Scenario 3.2",
        description="Cross platform, same topic (2-1)"
    )

    for test_platform in PLATFORMS:
        train_platforms = [p for p in PLATFORMS if p != test_platform]

        for video in VIDEOS:
            train_names = "".join(PLATFORM_NAMES[p] for p in train_platforms)
            sub_exp = SubExperiment(
                name=f"Scenario3.2_{train_names} vs {PLATFORM_NAMES[test_platform]}_Video{video}",
                scenario_id="3.2",
                train_filter={
                    "platform_id": train_platforms,
                    "video_id": [video],
                    "session_id": ALL_SESSIONS
                },
                test_filter={
                    "platform_id": [test_platform],
                    "video_id": [video],
                    "session_id": ALL_SESSIONS
                }
            )
            scenario.sub_experiments.append(sub_exp)

    return scenario


def generate_scenario_4_1() -> Scenario:
    """
    Scenario 4.1: Cross platform, cross topic (1-1)

    Train on 1 platform with 1 video, test on different platform with different video.
    Platform != Platform AND Video != Video.
    All sessions used.

    Sub-experiments: Based on the specification (9 combinations)
    """
    scenario = Scenario(
        id="4.1",
        name="Scenario 4.1",
        description="Cross platform, cross topic (1-1)"
    )

    # From specification:
    # F,V1 -> I,V2 | F,V1 -> I,V3 | F,V2 -> I,V3
    # F,V1 -> T,V2 | F,V1 -> T,V3 | F,V2 -> T,V3
    # I,V1 -> T,V2 | I,V1 -> T,V3 | I,V2 -> T,V3

    experiment_specs = [
        # F -> I combinations
        ((1, 1), (2, 2)),
        ((1, 1), (2, 3)),
        ((1, 2), (2, 3)),
        # F -> T combinations
        ((1, 1), (3, 2)),
        ((1, 1), (3, 3)),
        ((1, 2), (3, 3)),
        # I -> T combinations
        ((2, 1), (3, 2)),
        ((2, 1), (3, 3)),
        ((2, 2), (3, 3)),
    ]

    for (train_p, train_v), (test_p, test_v) in experiment_specs:
        sub_exp = SubExperiment(
            name=f"Scenario4.1_{PLATFORM_NAMES[train_p]}_Video{train_v} vs {PLATFORM_NAMES[test_p]}_Video{test_v}",
            scenario_id="4.1",
            train_filter={
                "platform_id": [train_p],
                "video_id": [train_v],
                "session_id": ALL_SESSIONS
            },
            test_filter={
                "platform_id": [test_p],
                "video_id": [test_v],
                "session_id": ALL_SESSIONS
            }
        )
        scenario.sub_experiments.append(sub_exp)

    return scenario


def generate_scenario_4_2() -> Scenario:
    """
    Scenario 4.2: Cross platform, cross topic (2-1)

    Train on 2 platforms with 1 video, test on remaining platform with different video.
    All sessions used.

    Sub-experiments: 9 based on specification
    """
    scenario = Scenario(
        id="4.2",
        name="Scenario 4.2",
        description="Cross platform, cross topic (2-1)"
    )

    # From specification:
    # F+I,V1 -> T,V2 | F+I,V2 -> T,V3 | F+I,V3 -> T,V1
    # F+T,V1 -> I,V2 | F+I,V2 -> I,V3 | F+I,V3 -> I,V1
    # I+T,V1 -> F,V2 | I+T,V2 -> F,V3 | I+T,V3 -> F,V1

    experiment_specs = [
        # Train F+I, Test T
        (([1, 2], 1), (3, 2)),
        (([1, 2], 2), (3, 3)),
        (([1, 2], 3), (3, 1)),
        # Train F+T, Test I
        (([1, 3], 1), (2, 2)),
        (([1, 2], 2), (2, 3)),  # Note: spec shows F+I here, keeping as-is from original
        (([1, 2], 3), (2, 1)),  # Note: spec shows F+I here, keeping as-is from original
        # Train I+T, Test F
        (([2, 3], 1), (1, 2)),
        (([2, 3], 2), (1, 3)),
        (([2, 3], 3), (1, 1)),
    ]

    for (train_ps, train_v), (test_p, test_v) in experiment_specs:
        train_names = "".join(PLATFORM_NAMES[p] for p in train_ps)
        sub_exp = SubExperiment(
            name=f"Scenario4.2_{train_names}_Video{train_v} vs {PLATFORM_NAMES[test_p]}_Video{test_v}",
            scenario_id="4.2",
            train_filter={
                "platform_id": train_ps,
                "video_id": [train_v],
                "session_id": ALL_SESSIONS
            },
            test_filter={
                "platform_id": [test_p],
                "video_id": [test_v],
                "session_id": ALL_SESSIONS
            }
        )
        scenario.sub_experiments.append(sub_exp)

    return scenario


def generate_all_scenarios() -> Dict[str, Scenario]:
    """Generate all scenarios and return as a dictionary keyed by scenario ID."""
    scenarios = {
        "1": generate_scenario_1(),
        "2": generate_scenario_2(),
        "3.1": generate_scenario_3_1(),
        "3.2": generate_scenario_3_2(),
        "4.1": generate_scenario_4_1(),
        "4.2": generate_scenario_4_2(),
    }
    return scenarios


def get_scenario_by_id(scenario_id: str) -> Scenario:
    """Get a specific scenario by its ID."""
    generators = {
        "1": generate_scenario_1,
        "2": generate_scenario_2,
        "3.1": generate_scenario_3_1,
        "3.2": generate_scenario_3_2,
        "4.1": generate_scenario_4_1,
        "4.2": generate_scenario_4_2,
    }

    if scenario_id not in generators:
        raise ValueError(f"Unknown scenario ID: {scenario_id}. Valid IDs: {list(generators.keys())}")

    return generators[scenario_id]()


def get_all_sub_experiments() -> List[SubExperiment]:
    """Get a flat list of all sub-experiments across all scenarios."""
    all_scenarios = generate_all_scenarios()
    all_sub_experiments = []
    for scenario in all_scenarios.values():
        all_sub_experiments.extend(scenario.sub_experiments)
    return all_sub_experiments


def print_scenario_summary():
    """Print a summary of all scenarios and their sub-experiments."""
    scenarios = generate_all_scenarios()
    total = 0

    print("=" * 70)
    print("SCENARIO SUMMARY")
    print("=" * 70)

    for scenario_id, scenario in scenarios.items():
        print(f"\n{scenario.name}: {scenario.description}")
        print(f"  Sub-experiments: {len(scenario)}")
        total += len(scenario)

        # Show first few examples
        for i, sub_exp in enumerate(scenario.sub_experiments[:3]):
            train_desc = _format_filter_name(sub_exp.train_filter)
            test_desc = _format_filter_name(sub_exp.test_filter)
            print(f"    [{i+1}] {sub_exp.name}: {train_desc} -> {test_desc}")

        if len(scenario) > 3:
            print(f"    ... and {len(scenario) - 3} more")

    print(f"\n{'=' * 70}")
    print(f"TOTAL SUB-EXPERIMENTS: {total}")
    print("=" * 70)


if __name__ == "__main__":
    # Print summary when run directly
    print_scenario_summary()
