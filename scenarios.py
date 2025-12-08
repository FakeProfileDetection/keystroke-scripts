#!/usr/bin/env python3
"""
scenarios.py - Scenario definitions and experiment generators for keystroke biometrics.

This module defines the 8 experimental scenarios and generates all sub-experiments
for each scenario. Results are aggregated at the scenario level.

Scenarios (from SCENARIO_REFERENCE.md):
    1.1: Same platform, same topic (S1 → S2)
    1.2: Same platform, same topic (S2 → S1)
    2.1: Cross platform, same topic (S1 → S2)
    2.2: Cross platform, same topic (S2 → S1)
    3.1: Cross platform, same topic (1-1, both sessions)
    3.2: Cross platform, same topic (2-1, both sessions)
    4.1: Cross platform, cross topic (1-1, both sessions)
    4.2: Cross platform, cross topic (2-1, both sessions)

Notation:
    P = Platform (F=1=Facebook, I=2=Instagram, T=3=Twitter)
    V = Video/Topic (1, 2, 3)
    S = Session (1, 2)

Total sub-experiments: 90
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
    train_filter: Dict[str, List[int]]  # e.g., {"platform_id": [1], "video_id": [1], "session_id": [1]}
    test_filter: Dict[str, List[int]]   # e.g., {"platform_id": [2], "video_id": [1], "session_id": [2]}

    # For template output formatting
    train_notation: str = ""  # e.g., "[PF, V1, S1]"
    test_notation: str = ""   # e.g., "[PF, V1, S2]"

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
    train_samples_per_user: int = 1
    test_samples_per_user: int = 1

    def __len__(self):
        return len(self.sub_experiments)


def _format_notation(platform_ids: List[int], video_ids: List[int], session_ids: List[int]) -> str:
    """Format filter as notation like [Pi, Vj, Sk] i={F}, j={1}, k={1}."""
    p_str = ",".join(PLATFORM_NAMES[p] for p in platform_ids)
    v_str = ",".join(str(v) for v in video_ids)
    s_str = ",".join(str(s) for s in session_ids)

    if len(session_ids) == 2:
        s_part = "k={1,2}"
    else:
        s_part = f"k={{{s_str}}}"

    return f"[Pi, Vj, Sk] i={{{p_str}}}, j={{{v_str}}}, {s_part}"


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


# =============================================================================
# Scenario 1.1: Same Platform, Same Topic (S1 → S2)
# =============================================================================
def generate_scenario_1_1() -> Scenario:
    """
    Scenario 1.1: Same platform, same topic (S1 → S2)

    Train on session 1, test on session 2. Same platform and video.

    Sub-experiments: 3 platforms × 3 videos = 9
    """
    scenario = Scenario(
        id="1.1",
        name="Scenario 1.1",
        description="Same platform, same topic (S1 → S2)",
        train_samples_per_user=1,
        test_samples_per_user=1
    )

    for platform in PLATFORMS:
        for video in VIDEOS:
            sub_exp = SubExperiment(
                name=f"Scenario1.1_{PLATFORM_NAMES[platform]}_{VIDEO_NAMES[video]}_S1_to_S2",
                scenario_id="1.1",
                train_filter={
                    "platform_id": [platform],
                    "video_id": [video],
                    "session_id": [1]
                },
                test_filter={
                    "platform_id": [platform],
                    "video_id": [video],
                    "session_id": [2]
                },
                train_notation=_format_notation([platform], [video], [1]),
                test_notation=_format_notation([platform], [video], [2])
            )
            scenario.sub_experiments.append(sub_exp)

    return scenario


# =============================================================================
# Scenario 1.2: Same Platform, Same Topic (S2 → S1)
# =============================================================================
def generate_scenario_1_2() -> Scenario:
    """
    Scenario 1.2: Same platform, same topic (S2 → S1)

    Train on session 2, test on session 1. Same platform and video.

    Sub-experiments: 3 platforms × 3 videos = 9
    """
    scenario = Scenario(
        id="1.2",
        name="Scenario 1.2",
        description="Same platform, same topic (S2 → S1)",
        train_samples_per_user=1,
        test_samples_per_user=1
    )

    for platform in PLATFORMS:
        for video in VIDEOS:
            sub_exp = SubExperiment(
                name=f"Scenario1.2_{PLATFORM_NAMES[platform]}_{VIDEO_NAMES[video]}_S2_to_S1",
                scenario_id="1.2",
                train_filter={
                    "platform_id": [platform],
                    "video_id": [video],
                    "session_id": [2]
                },
                test_filter={
                    "platform_id": [platform],
                    "video_id": [video],
                    "session_id": [1]
                },
                train_notation=_format_notation([platform], [video], [2]),
                test_notation=_format_notation([platform], [video], [1])
            )
            scenario.sub_experiments.append(sub_exp)

    return scenario


# =============================================================================
# Scenario 2.1: Cross Platform, Same Topic (S1 → S2)
# =============================================================================
def generate_scenario_2_1() -> Scenario:
    """
    Scenario 2.1: Cross platform, same topic (S1 → S2)

    Train on platform X session 1, test on platform Y session 2.
    Same video, all 6 platform pair directions.

    Sub-experiments: 6 platform pairs × 3 videos = 18
    """
    scenario = Scenario(
        id="2.1",
        name="Scenario 2.1",
        description="Cross platform, same topic (S1 → S2)",
        train_samples_per_user=1,
        test_samples_per_user=1
    )

    # All 6 directed platform pairs
    platform_pairs = [
        (1, 2),  # F → I
        (1, 3),  # F → T
        (2, 3),  # I → T
        (2, 1),  # I → F
        (3, 1),  # T → F
        (3, 2),  # T → I
    ]

    for train_p, test_p in platform_pairs:
        for video in VIDEOS:
            sub_exp = SubExperiment(
                name=f"Scenario2.1_{PLATFORM_NAMES[train_p]}_to_{PLATFORM_NAMES[test_p]}_{VIDEO_NAMES[video]}_S1_to_S2",
                scenario_id="2.1",
                train_filter={
                    "platform_id": [train_p],
                    "video_id": [video],
                    "session_id": [1]
                },
                test_filter={
                    "platform_id": [test_p],
                    "video_id": [video],
                    "session_id": [2]
                },
                train_notation=_format_notation([train_p], [video], [1]),
                test_notation=_format_notation([test_p], [video], [2])
            )
            scenario.sub_experiments.append(sub_exp)

    return scenario


# =============================================================================
# Scenario 2.2: Cross Platform, Same Topic (S2 → S1)
# =============================================================================
def generate_scenario_2_2() -> Scenario:
    """
    Scenario 2.2: Cross platform, same topic (S2 → S1)

    Train on platform X session 2, test on platform Y session 1.
    Same video, all 6 platform pair directions.

    Sub-experiments: 6 platform pairs × 3 videos = 18
    """
    scenario = Scenario(
        id="2.2",
        name="Scenario 2.2",
        description="Cross platform, same topic (S2 → S1)",
        train_samples_per_user=1,
        test_samples_per_user=1
    )

    # All 6 directed platform pairs
    platform_pairs = [
        (1, 2),  # F → I
        (1, 3),  # F → T
        (2, 3),  # I → T
        (2, 1),  # I → F
        (3, 1),  # T → F
        (3, 2),  # T → I
    ]

    for train_p, test_p in platform_pairs:
        for video in VIDEOS:
            sub_exp = SubExperiment(
                name=f"Scenario2.2_{PLATFORM_NAMES[train_p]}_to_{PLATFORM_NAMES[test_p]}_{VIDEO_NAMES[video]}_S2_to_S1",
                scenario_id="2.2",
                train_filter={
                    "platform_id": [train_p],
                    "video_id": [video],
                    "session_id": [2]
                },
                test_filter={
                    "platform_id": [test_p],
                    "video_id": [video],
                    "session_id": [1]
                },
                train_notation=_format_notation([train_p], [video], [2]),
                test_notation=_format_notation([test_p], [video], [1])
            )
            scenario.sub_experiments.append(sub_exp)

    return scenario


# =============================================================================
# Scenario 3.1: Cross Platform, Same Topic (1→1, Both Sessions)
# =============================================================================
def generate_scenario_3_1() -> Scenario:
    """
    Scenario 3.1: Cross platform, same topic (1→1, both sessions)

    Train on 1 platform, test on 1 different platform.
    Same video, both sessions.

    Sub-experiments: 3 platform pairs × 3 videos = 9
    (Only F→I, F→T, I→T directions)
    """
    scenario = Scenario(
        id="3.1",
        name="Scenario 3.1",
        description="Cross platform, same topic (1→1, both sessions)",
        train_samples_per_user=2,
        test_samples_per_user=2
    )

    # Platform pairs (ordered: train -> test)
    # From SCENARIO_REFERENCE: F→I, F→T, I→T only
    platform_pairs = [
        (1, 2),  # F → I
        (1, 3),  # F → T
        (2, 3),  # I → T
    ]

    for train_p, test_p in platform_pairs:
        for video in VIDEOS:
            sub_exp = SubExperiment(
                name=f"Scenario3.1_{PLATFORM_NAMES[train_p]}_to_{PLATFORM_NAMES[test_p]}_{VIDEO_NAMES[video]}",
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
                },
                train_notation=_format_notation([train_p], [video], ALL_SESSIONS),
                test_notation=_format_notation([test_p], [video], ALL_SESSIONS)
            )
            scenario.sub_experiments.append(sub_exp)

    return scenario


# =============================================================================
# Scenario 3.2: Cross Platform, Same Topic (2→1, Both Sessions)
# =============================================================================
def generate_scenario_3_2() -> Scenario:
    """
    Scenario 3.2: Cross platform, same topic (2→1, both sessions)

    Train on 2 platforms, test on 1 remaining platform.
    Same video, both sessions. Leave-one-platform-out.

    Sub-experiments: 3 test platforms × 3 videos = 9
    """
    scenario = Scenario(
        id="3.2",
        name="Scenario 3.2",
        description="Cross platform, same topic (2→1, both sessions)",
        train_samples_per_user=4,
        test_samples_per_user=2
    )

    # Leave-one-platform-out: train on 2, test on 1
    # From SCENARIO_REFERENCE:
    # F+I → T, I+T → F, T+F → I
    leave_one_out = [
        ([1, 2], 3),  # F+I → T
        ([2, 3], 1),  # I+T → F
        ([3, 1], 2),  # T+F → I
    ]

    for train_ps, test_p in leave_one_out:
        for video in VIDEOS:
            train_names = "+".join(PLATFORM_NAMES[p] for p in train_ps)
            sub_exp = SubExperiment(
                name=f"Scenario3.2_{train_names}_to_{PLATFORM_NAMES[test_p]}_{VIDEO_NAMES[video]}",
                scenario_id="3.2",
                train_filter={
                    "platform_id": train_ps,
                    "video_id": [video],
                    "session_id": ALL_SESSIONS
                },
                test_filter={
                    "platform_id": [test_p],
                    "video_id": [video],
                    "session_id": ALL_SESSIONS
                },
                train_notation=_format_notation(train_ps, [video], ALL_SESSIONS),
                test_notation=_format_notation([test_p], [video], ALL_SESSIONS)
            )
            scenario.sub_experiments.append(sub_exp)

    return scenario


# =============================================================================
# Scenario 4.1: Cross Platform, Cross Topic (1→1, Both Sessions)
# =============================================================================
def generate_scenario_4_1() -> Scenario:
    """
    Scenario 4.1: Cross platform, cross topic (1→1, both sessions)

    Train on 1 platform with 1 video, test on different platform with different video.
    Both sessions used.

    Sub-experiments: 9 (from SCENARIO_REFERENCE)
    """
    scenario = Scenario(
        id="4.1",
        name="Scenario 4.1",
        description="Cross platform, cross topic (1→1, both sessions)",
        train_samples_per_user=2,
        test_samples_per_user=2
    )

    # From SCENARIO_REFERENCE.md:
    # (train_platform, train_video) -> (test_platform, test_video)
    experiment_specs = [
        ((1, 1), (2, 2)),  # F,V1 → I,V2
        ((1, 1), (2, 3)),  # F,V1 → I,V3
        ((1, 2), (2, 3)),  # F,V2 → I,V3
        ((1, 1), (3, 2)),  # F,V1 → T,V2
        ((1, 1), (3, 3)),  # F,V1 → T,V3
        ((1, 2), (3, 3)),  # F,V2 → T,V3
        ((2, 1), (3, 2)),  # I,V1 → T,V2
        ((2, 1), (3, 3)),  # I,V1 → T,V3
        ((2, 2), (3, 3)),  # I,V2 → T,V3
    ]

    for (train_p, train_v), (test_p, test_v) in experiment_specs:
        sub_exp = SubExperiment(
            name=f"Scenario4.1_{PLATFORM_NAMES[train_p]}_{VIDEO_NAMES[train_v]}_to_{PLATFORM_NAMES[test_p]}_{VIDEO_NAMES[test_v]}",
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
            },
            train_notation=_format_notation([train_p], [train_v], ALL_SESSIONS),
            test_notation=_format_notation([test_p], [test_v], ALL_SESSIONS)
        )
        scenario.sub_experiments.append(sub_exp)

    return scenario


# =============================================================================
# Scenario 4.2: Cross Platform, Cross Topic (2→1, Both Sessions)
# =============================================================================
def generate_scenario_4_2() -> Scenario:
    """
    Scenario 4.2: Cross platform, cross topic (2→1, both sessions)

    Train on 2 platforms with 1 video, test on remaining platform with different video.
    Both sessions used.

    Sub-experiments: 9 (from SCENARIO_REFERENCE)
    """
    scenario = Scenario(
        id="4.2",
        name="Scenario 4.2",
        description="Cross platform, cross topic (2→1, both sessions)",
        train_samples_per_user=4,
        test_samples_per_user=2
    )

    # From SCENARIO_REFERENCE.md:
    # (train_platforms, train_video) -> (test_platform, test_video)
    experiment_specs = [
        (([1, 2], 1), (3, 2)),  # F+I,V1 → T,V2
        (([1, 2], 2), (3, 3)),  # F+I,V2 → T,V3
        (([1, 2], 3), (3, 1)),  # F+I,V3 → T,V1
        (([1, 3], 1), (2, 2)),  # F+T,V1 → I,V2
        (([1, 3], 2), (2, 3)),  # F+T,V2 → I,V3
        (([1, 3], 3), (2, 1)),  # F+T,V3 → I,V1
        (([2, 3], 1), (1, 2)),  # I+T,V1 → F,V2
        (([2, 3], 2), (1, 3)),  # I+T,V2 → F,V3
        (([2, 3], 3), (1, 1)),  # I+T,V3 → F,V1
    ]

    for (train_ps, train_v), (test_p, test_v) in experiment_specs:
        train_names = "+".join(PLATFORM_NAMES[p] for p in train_ps)
        sub_exp = SubExperiment(
            name=f"Scenario4.2_{train_names}_{VIDEO_NAMES[train_v]}_to_{PLATFORM_NAMES[test_p]}_{VIDEO_NAMES[test_v]}",
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
            },
            train_notation=_format_notation(train_ps, [train_v], ALL_SESSIONS),
            test_notation=_format_notation([test_p], [test_v], ALL_SESSIONS)
        )
        scenario.sub_experiments.append(sub_exp)

    return scenario


# =============================================================================
# Utility Functions
# =============================================================================

def generate_all_scenarios() -> Dict[str, Scenario]:
    """Generate all scenarios and return as a dictionary keyed by scenario ID."""
    scenarios = {
        "1.1": generate_scenario_1_1(),
        "1.2": generate_scenario_1_2(),
        "2.1": generate_scenario_2_1(),
        "2.2": generate_scenario_2_2(),
        "3.1": generate_scenario_3_1(),
        "3.2": generate_scenario_3_2(),
        "4.1": generate_scenario_4_1(),
        "4.2": generate_scenario_4_2(),
    }
    return scenarios


def get_scenario_by_id(scenario_id: str) -> Scenario:
    """Get a specific scenario by its ID."""
    generators = {
        "1.1": generate_scenario_1_1,
        "1.2": generate_scenario_1_2,
        "2.1": generate_scenario_2_1,
        "2.2": generate_scenario_2_2,
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

    print("=" * 80)
    print("SCENARIO SUMMARY (from SCENARIO_REFERENCE.md)")
    print("=" * 80)

    for scenario_id, scenario in scenarios.items():
        print(f"\n{scenario.name}: {scenario.description}")
        print(f"  Sub-experiments: {len(scenario)}")
        print(f"  Train samples/user: {scenario.train_samples_per_user}")
        print(f"  Test samples/user: {scenario.test_samples_per_user}")
        total += len(scenario)

        # Show first few examples
        for i, sub_exp in enumerate(scenario.sub_experiments[:3]):
            print(f"    [{i+1}] {sub_exp.name}")
            print(f"        Train: {sub_exp.train_notation}")
            print(f"        Test:  {sub_exp.test_notation}")

        if len(scenario) > 3:
            print(f"    ... and {len(scenario) - 3} more")

    print(f"\n{'=' * 80}")
    print(f"TOTAL SUB-EXPERIMENTS: {total}")
    print("=" * 80)


if __name__ == "__main__":
    # Print summary when run directly
    print_scenario_summary()
