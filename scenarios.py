#!/usr/bin/env python3
"""
scenarios.py - Scenario definitions for keystroke biometrics experiments.

Based on scenario-template-14Dec2025.tsv

Scenarios:
    1.1: Same platform, same topic (S1 -> S2)
    1.2: Same platform, same topic (S2 -> S1)
    2.1: Cross platform, same topic (1-1) (S1 -> S2)
    2.2: Cross platform, same topic (1-1) (S2 -> S1)
    3.1: Cross platform, same topic (1-1) (both sessions)
    3.2: Cross platform, same topic (2-1) (both sessions)
    4.1: Cross platform, cross topic (1-1) (both sessions)
    4.2: Cross platform, cross topic (2-1) (both sessions)
    5.1: Same platform, cross topic (S1 -> S2)
    5.2: Same platform, cross topic (S2 -> S1)

Notation:
    P = Platform (F=1=Facebook, I=2=Instagram, T=3=Twitter)
    V = Video/Topic (1, 2, 3)
    S = Session (1, 2)

Total sub-experiments: 171
"""

from typing import List, Dict, Tuple
from dataclasses import dataclass, field


# Constants
PLATFORMS = [1, 2, 3]  # F=1, I=2, T=3
VIDEOS = [1, 2, 3]
ALL_SESSIONS = [1, 2]

PLATFORM_NAMES = {1: "F", 2: "I", 3: "T"}


@dataclass
class SubExperiment:
    """Represents a single sub-experiment within a scenario."""
    name: str
    scenario_id: str
    train_filter: Dict[str, List[int]]
    test_filter: Dict[str, List[int]]
    train_notation: str = ""
    test_notation: str = ""

    def get_train_mask_conditions(self) -> List[Tuple[str, List[int]]]:
        return [(col, vals) for col, vals in self.train_filter.items()]

    def get_test_mask_conditions(self) -> List[Tuple[str, List[int]]]:
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


def _notation(platforms: List[int], videos: List[int], sessions: List[int]) -> str:
    """Format as [Pi, Vj, Sk] i={...}, j={...}, k={...}."""
    p_str = ", ".join(PLATFORM_NAMES[p] for p in platforms)
    v_str = ", ".join(str(v) for v in videos)
    if len(sessions) == 2:
        s_str = "1 ,2"
    else:
        s_str = str(sessions[0])
    return f"[Pi, Vj, Sk] i={{{p_str}}}, j ={{{v_str}}}, k={{{s_str}}}"


# =============================================================================
# Scenario 1.1: Same Platform, Same Topic (S1 -> S2)
# =============================================================================
def generate_scenario_1_1() -> Scenario:
    """9 sub-experiments: 3 platforms x 3 videos, train S1, test S2."""
    scenario = Scenario(
        id="1.1", name="Scenario 1.1",
        description="Same platform, same topic",
        train_samples_per_user=1, test_samples_per_user=1
    )
    for p in PLATFORMS:
        for v in VIDEOS:
            scenario.sub_experiments.append(SubExperiment(
                name=f"S1.1_{PLATFORM_NAMES[p]}_{v}_S1toS2",
                scenario_id="1.1",
                train_filter={"platform_id": [p], "video_id": [v], "session_id": [1]},
                test_filter={"platform_id": [p], "video_id": [v], "session_id": [2]},
                train_notation=_notation([p], [v], [1]),
                test_notation=_notation([p], [v], [2])
            ))
    return scenario


# =============================================================================
# Scenario 1.2: Same Platform, Same Topic (S2 -> S1)
# =============================================================================
def generate_scenario_1_2() -> Scenario:
    """9 sub-experiments: 3 platforms x 3 videos, train S2, test S1."""
    scenario = Scenario(
        id="1.2", name="Scenario 1.2",
        description="Same platform, same topic",
        train_samples_per_user=1, test_samples_per_user=1
    )
    for p in PLATFORMS:
        for v in VIDEOS:
            scenario.sub_experiments.append(SubExperiment(
                name=f"S1.2_{PLATFORM_NAMES[p]}_{v}_S2toS1",
                scenario_id="1.2",
                train_filter={"platform_id": [p], "video_id": [v], "session_id": [2]},
                test_filter={"platform_id": [p], "video_id": [v], "session_id": [1]},
                train_notation=_notation([p], [v], [2]),
                test_notation=_notation([p], [v], [1])
            ))
    return scenario


# =============================================================================
# Scenario 2.1: Cross Platform, Same Topic (1-1) (S1 -> S2)
# =============================================================================
def generate_scenario_2_1() -> Scenario:
    """18 sub-experiments: 6 platform pairs x 3 videos, train S1, test S2."""
    scenario = Scenario(
        id="2.1", name="Scenario 2.1",
        description="Cross platform, same topic (1-1) (controlled for dataset size)",
        train_samples_per_user=1, test_samples_per_user=1
    )
    # All 6 directed platform pairs
    pairs = [(1,2), (1,3), (2,3), (2,1), (3,1), (3,2)]
    for train_p, test_p in pairs:
        for v in VIDEOS:
            scenario.sub_experiments.append(SubExperiment(
                name=f"S2.1_{PLATFORM_NAMES[train_p]}to{PLATFORM_NAMES[test_p]}_{v}_S1toS2",
                scenario_id="2.1",
                train_filter={"platform_id": [train_p], "video_id": [v], "session_id": [1]},
                test_filter={"platform_id": [test_p], "video_id": [v], "session_id": [2]},
                train_notation=_notation([train_p], [v], [1]),
                test_notation=_notation([test_p], [v], [2])
            ))
    return scenario


# =============================================================================
# Scenario 2.2: Cross Platform, Same Topic (1-1) (S2 -> S1)
# =============================================================================
def generate_scenario_2_2() -> Scenario:
    """18 sub-experiments: 6 platform pairs x 3 videos, train S2, test S1."""
    scenario = Scenario(
        id="2.2", name="Scenario 2.2",
        description="Cross platform, same topic (1-1) (controlled for dataset size)",
        train_samples_per_user=1, test_samples_per_user=1
    )
    pairs = [(1,2), (1,3), (2,3), (2,1), (3,1), (3,2)]
    for train_p, test_p in pairs:
        for v in VIDEOS:
            scenario.sub_experiments.append(SubExperiment(
                name=f"S2.2_{PLATFORM_NAMES[train_p]}to{PLATFORM_NAMES[test_p]}_{v}_S2toS1",
                scenario_id="2.2",
                train_filter={"platform_id": [train_p], "video_id": [v], "session_id": [2]},
                test_filter={"platform_id": [test_p], "video_id": [v], "session_id": [1]},
                train_notation=_notation([train_p], [v], [2]),
                test_notation=_notation([test_p], [v], [1])
            ))
    return scenario


# =============================================================================
# Scenario 3.1: Cross Platform, Same Topic (1-1) (Both Sessions)
# =============================================================================
def generate_scenario_3_1() -> Scenario:
    """18 sub-experiments: 6 platform pairs x 3 videos, both sessions."""
    scenario = Scenario(
        id="3.1", name="Scenario 3.1",
        description="Cross platform, same topic (1-1)",
        train_samples_per_user=2, test_samples_per_user=2
    )
    # All 6 directed platform pairs (both directions)
    pairs = [(1,2), (1,3), (2,3), (2,1), (3,1), (3,2)]
    for train_p, test_p in pairs:
        for v in VIDEOS:
            scenario.sub_experiments.append(SubExperiment(
                name=f"S3.1_{PLATFORM_NAMES[train_p]}to{PLATFORM_NAMES[test_p]}_{v}",
                scenario_id="3.1",
                train_filter={"platform_id": [train_p], "video_id": [v], "session_id": ALL_SESSIONS},
                test_filter={"platform_id": [test_p], "video_id": [v], "session_id": ALL_SESSIONS},
                train_notation=_notation([train_p], [v], ALL_SESSIONS),
                test_notation=_notation([test_p], [v], ALL_SESSIONS)
            ))
    return scenario


# =============================================================================
# Scenario 3.2: Cross Platform, Same Topic (2-1) (Both Sessions)
# =============================================================================
def generate_scenario_3_2() -> Scenario:
    """9 sub-experiments: 3 leave-one-out x 3 videos, both sessions."""
    scenario = Scenario(
        id="3.2", name="Scenario 3.2",
        description="Cross platform, same topic (2-1)",
        train_samples_per_user=4, test_samples_per_user=2
    )
    # Leave-one-platform-out: F+I->T, I+T->F, T+F->I
    configs = [([1,2], 3), ([2,3], 1), ([3,1], 2)]
    for train_ps, test_p in configs:
        for v in VIDEOS:
            p_names = ", ".join(PLATFORM_NAMES[p] for p in train_ps)
            scenario.sub_experiments.append(SubExperiment(
                name=f"S3.2_{p_names}to{PLATFORM_NAMES[test_p]}_{v}",
                scenario_id="3.2",
                train_filter={"platform_id": train_ps, "video_id": [v], "session_id": ALL_SESSIONS},
                test_filter={"platform_id": [test_p], "video_id": [v], "session_id": ALL_SESSIONS},
                train_notation=_notation(train_ps, [v], ALL_SESSIONS),
                test_notation=_notation([test_p], [v], ALL_SESSIONS)
            ))
    return scenario


# =============================================================================
# Scenario 4.1: Cross Platform, Cross Topic (1-1) (Both Sessions)
# =============================================================================
def generate_scenario_4_1() -> Scenario:
    """36 sub-experiments: all cross-platform cross-topic combinations."""
    scenario = Scenario(
        id="4.1", name="Scenario 4.1",
        description="Cross platform, cross topic (1-1)",
        train_samples_per_user=2, test_samples_per_user=2
    )
    # For each platform, for each video, test on all other platforms with all other videos
    for train_p in PLATFORMS:
        for train_v in VIDEOS:
            for test_p in PLATFORMS:
                if test_p == train_p:
                    continue
                for test_v in VIDEOS:
                    if test_v == train_v:
                        continue
                    scenario.sub_experiments.append(SubExperiment(
                        name=f"S4.1_{PLATFORM_NAMES[train_p]}{train_v}to{PLATFORM_NAMES[test_p]}{test_v}",
                        scenario_id="4.1",
                        train_filter={"platform_id": [train_p], "video_id": [train_v], "session_id": ALL_SESSIONS},
                        test_filter={"platform_id": [test_p], "video_id": [test_v], "session_id": ALL_SESSIONS},
                        train_notation=_notation([train_p], [train_v], ALL_SESSIONS),
                        test_notation=_notation([test_p], [test_v], ALL_SESSIONS)
                    ))
    return scenario


# =============================================================================
# Scenario 4.2: Cross Platform, Cross Topic (2-1) (Both Sessions)
# =============================================================================
def generate_scenario_4_2() -> Scenario:
    """18 sub-experiments: 2 train platforms, 1 test platform, cross topic.

    Order matches template: forward rotation (+1) then backward rotation (-1).
    Video pairs per platform config: V1→V2, V2→V3, V3→V1, V1→V3, V2→V1, V3→V2
    """
    scenario = Scenario(
        id="4.2", name="Scenario 4.2",
        description="Cross platform, cross topic (2-1)",
        train_samples_per_user=4, test_samples_per_user=2
    )
    # Video pair order from template: forward rotation then backward rotation
    # Forward: (V1,V2), (V2,V3), (V3,V1)  - test = (train % 3) + 1
    # Backward: (V1,V3), (V2,V1), (V3,V2) - test = ((train + 1) % 3) + 1
    video_pairs = [
        (1, 2), (2, 3), (3, 1),  # forward rotation
        (1, 3), (2, 1), (3, 2),  # backward rotation
    ]
    # Platform configs from template: F+I->T, F+T->I, I+T->F
    configs = [
        ([1,2], 3),  # F+I -> T
        ([1,3], 2),  # F+T -> I
        ([2,3], 1),  # I+T -> F
    ]
    for train_ps, test_p in configs:
        for train_v, test_v in video_pairs:
            p_names = ", ".join(PLATFORM_NAMES[p] for p in train_ps)
            scenario.sub_experiments.append(SubExperiment(
                name=f"S4.2_{p_names}{train_v}to{PLATFORM_NAMES[test_p]}{test_v}",
                scenario_id="4.2",
                train_filter={"platform_id": train_ps, "video_id": [train_v], "session_id": ALL_SESSIONS},
                test_filter={"platform_id": [test_p], "video_id": [test_v], "session_id": ALL_SESSIONS},
                train_notation=_notation(train_ps, [train_v], ALL_SESSIONS),
                test_notation=_notation([test_p], [test_v], ALL_SESSIONS)
            ))
    return scenario


# =============================================================================
# Scenario 5.1: Same Platform, Cross Topic (S1 -> S2)
# =============================================================================
def generate_scenario_5_1() -> Scenario:
    """18 sub-experiments: 3 platforms x 6 video pairs, train S1, test S2.

    Order matches template: forward rotation (+1) then backward rotation (-1).
    Video pairs per platform: V1→V2, V2→V3, V3→V1, V2→V1, V3→V2, V1→V3
    """
    scenario = Scenario(
        id="5.1", name="Scenario 5.1",
        description="Same platform, cross topic",
        train_samples_per_user=1, test_samples_per_user=1
    )
    # Video pair order from template: forward rotation then backward rotation
    video_pairs = [
        (1, 2), (2, 3), (3, 1),  # forward rotation
        (2, 1), (3, 2), (1, 3),  # backward rotation
    ]
    for p in PLATFORMS:
        for train_v, test_v in video_pairs:
            scenario.sub_experiments.append(SubExperiment(
                name=f"S5.1_{PLATFORM_NAMES[p]}_{train_v}to{test_v}_S1toS2",
                scenario_id="5.1",
                train_filter={"platform_id": [p], "video_id": [train_v], "session_id": [1]},
                test_filter={"platform_id": [p], "video_id": [test_v], "session_id": [2]},
                train_notation=_notation([p], [train_v], [1]),
                test_notation=_notation([p], [test_v], [2])
            ))
    return scenario


# =============================================================================
# Scenario 5.2: Same Platform, Cross Topic (S2 -> S1)
# =============================================================================
def generate_scenario_5_2() -> Scenario:
    """18 sub-experiments: 3 platforms x 6 video pairs, train S2, test S1.

    Order matches template: backward rotation first, then forward rotation.
    Video pairs per platform: V2→V1, V3→V2, V1→V3, V1→V2, V2→V3, V3→V1
    """
    scenario = Scenario(
        id="5.2", name="Scenario 5.2",
        description="Same platform, cross topic",
        train_samples_per_user=1, test_samples_per_user=1
    )
    # Video pair order from template: backward rotation then forward rotation
    video_pairs = [
        (2, 1), (3, 2), (1, 3),  # backward rotation
        (1, 2), (2, 3), (3, 1),  # forward rotation
    ]
    for p in PLATFORMS:
        for train_v, test_v in video_pairs:
            scenario.sub_experiments.append(SubExperiment(
                name=f"S5.2_{PLATFORM_NAMES[p]}_{train_v}to{test_v}_S2toS1",
                scenario_id="5.2",
                train_filter={"platform_id": [p], "video_id": [train_v], "session_id": [2]},
                test_filter={"platform_id": [p], "video_id": [test_v], "session_id": [1]},
                train_notation=_notation([p], [train_v], [2]),
                test_notation=_notation([p], [test_v], [1])
            ))
    return scenario


# =============================================================================
# Utility Functions
# =============================================================================

def generate_all_scenarios() -> Dict[str, Scenario]:
    """Generate all scenarios."""
    return {
        "1.1": generate_scenario_1_1(),
        "1.2": generate_scenario_1_2(),
        "2.1": generate_scenario_2_1(),
        "2.2": generate_scenario_2_2(),
        "3.1": generate_scenario_3_1(),
        "3.2": generate_scenario_3_2(),
        "4.1": generate_scenario_4_1(),
        "4.2": generate_scenario_4_2(),
        "5.1": generate_scenario_5_1(),
        "5.2": generate_scenario_5_2(),
    }


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
        "5.1": generate_scenario_5_1,
        "5.2": generate_scenario_5_2,
    }
    if scenario_id not in generators:
        raise ValueError(f"Unknown scenario ID: {scenario_id}. Valid: {list(generators.keys())}")
    return generators[scenario_id]()


def get_all_sub_experiments() -> List[SubExperiment]:
    """Get a flat list of all sub-experiments."""
    all_scenarios = generate_all_scenarios()
    return [se for s in all_scenarios.values() for se in s.sub_experiments]


def print_scenario_summary():
    """Print summary of all scenarios."""
    scenarios = generate_all_scenarios()
    total = 0
    print("=" * 70)
    print("SCENARIO SUMMARY (scenario-template-14Dec2025.tsv)")
    print("=" * 70)
    for sid, s in scenarios.items():
        print(f"\n{s.name}: {s.description}")
        print(f"  Sub-experiments: {len(s)}, Train: {s.train_samples_per_user}/user, Test: {s.test_samples_per_user}/user")
        total += len(s)
        for i, se in enumerate(s.sub_experiments[:2]):
            print(f"    [{i+1}] {se.train_notation} -> {se.test_notation}")
        if len(s) > 2:
            print(f"    ... and {len(s)-2} more")
    print(f"\n{'='*70}")
    print(f"TOTAL SUB-EXPERIMENTS: {total}")
    print("=" * 70)


if __name__ == "__main__":
    print_scenario_summary()
