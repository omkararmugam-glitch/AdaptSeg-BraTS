#!/usr/bin/env python
"""Run one project phase, or the whole analysis chain.

    python run_phase.py 1                 # one phase
    python run_phase.py 5 --n-patients 0  # phase options pass straight through
    python run_phase.py all               # phases 1-8 (9 trains, so it is opt-in)
    python run_phase.py list              # what each phase does and produces

Every phase is also runnable on its own:

    python -m phases.phase01_data_loading --help      (from inside src/)
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parent / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

PHASE_MODULES = {
    1: "phases.phase01_data_loading",
    2: "phases.phase02_integrity",
    3: "phases.phase03_intensity_analysis",
    4: "phases.phase04_cohort_distribution",
    5: "phases.phase05_quality_features",
    6: "phases.phase06_adaptive_selector",
    7: "phases.phase07_preprocessing_execution",
    8: "phases.phase08_model_selection",
    9: "phases.phase09_training_benchmark",
}

# Phases 1-8 are analysis and run in minutes. Phase 9 trains networks, so "all"
# stops before it rather than kicking off hours of GPU work unasked.
CHAIN = [1, 2, 3, 4, 5, 6, 7, 8]

PHASE_SUMMARY = {
    1: ("Data loading and first-patient inspection",
        "patient_inventory.csv, first_patient_volumes.csv"),
    2: ("Dataset integrity audit",
        "missing_modalities.csv, unreadable_files.csv, shape_compatibility.csv, "
        "segmentation_class_counts.csv, integrity_summary.csv"),
    3: ("Intensity analysis of a representative patient",
        "intensity_statistics.csv, intensity_findings.csv"),
    4: ("Cohort-wide intensity distribution analysis",
        "representative_subset.csv, per_patient_foreground_stats.csv, "
        "pooled_percentile_profile.csv, interpatient_variation.csv"),
    5: ("Image-quality feature extraction",
        "image_quality_features.csv, feature_correlations.csv, feature_variation.csv"),
    6: ("Adaptive preprocessing selector",
        "pipeline_catalog.csv, cohort_references.csv, preprocessing_selection.csv, "
        "selection_distribution.csv"),
    7: ("Adaptive preprocessing execution and validation",
        "execution_validation.csv, intensity_comparison.csv, "
        "pipeline_delta_summary.csv, segmentation_integrity.csv"),
    8: ("Patient splits, model registry and model-selection dataset",
        "patient_splits.csv, model_registry.csv, case_metadata.csv, "
        "model_selection_dataset.csv, case_level_performance.csv, case_best_model.csv"),
    9: ("Model training benchmark and held-out test evaluation",
        "training_history.csv, model_comparison.csv, test_results.csv, "
        "final_model_selection.csv"),
}

DEPENDENCIES = {6: [5], 7: [6], 8: [5, 6], 9: []}


def run_phase(number: int, argv) -> int:
    """Import a phase module and call its ``main``."""
    module = importlib.import_module(PHASE_MODULES[number])
    return int(module.main(list(argv)) or 0)


def _shared_option_strings() -> set:
    """The options accepted by every phase, taken from the common parser."""
    from phases import _common

    parser = _common.base_parser(0)
    options = set()
    for action in parser._actions:  # noqa: SLF001 - introspecting our own parser
        options.update(action.option_strings)
    return options


def _partition_chain_args(argv) -> tuple:
    """Split chain arguments into those every phase accepts and the rest."""
    shared_options = _shared_option_strings()
    shared, specific = [], []

    index = 0
    tokens = list(argv)
    while index < len(tokens):
        token = tokens[index]
        name = token.split("=", 1)[0]
        takes_value = (
            index + 1 < len(tokens)
            and not tokens[index + 1].startswith("-")
            and "=" not in token
        )

        if name in shared_options:
            shared.append(token)
            if takes_value and name in {"--dataset-path", "--seed"}:
                index += 1
                shared.append(tokens[index])
        elif token.startswith("-"):
            specific.append(token)
            if takes_value:
                index += 1
                specific.append(tokens[index])
        else:
            specific.append(token)
        index += 1

    return shared, specific


def print_listing() -> None:
    print("\nProject phases\n" + "=" * 72)
    for number, (title, artifacts) in PHASE_SUMMARY.items():
        needs = DEPENDENCIES.get(number) or []
        print("\nPhase " + str(number) + " - " + title)
        print("  module   : " + PHASE_MODULES[number])
        if needs:
            print("  requires : phase " + ", ".join(str(n) for n in needs))
        print("  artifacts: artifacts/phase" + format(number, "02d") + "/")
        for name in artifacts.split(", "):
            print("               " + name)
    print("\n" + "=" * 72)
    print("Run one phase:   python run_phase.py <n> [phase options]")
    print("Run the chain:   python run_phase.py all      (phases 1-8)")
    print("Phase help:      python run_phase.py <n> --help")
    print("Train (opt-in):  python run_phase.py 9 --fraction 0.5")


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(__doc__)
        print("Phases: " + ", ".join(str(n) for n in PHASE_MODULES) + ", all, list")
        return 0

    target, rest = argv[0], argv[1:]

    if target == "list":
        print_listing()
        return 0

    if target == "all":
        # Only the options every phase shares can be forwarded to all of them;
        # a phase-specific flag such as --skip-labels would make the others
        # abort on an unrecognised argument.
        shared, phase_specific = _partition_chain_args(rest)
        if phase_specific:
            print("These options are phase-specific and are not forwarded by 'all': "
                  + " ".join(phase_specific))
            print("Run that phase on its own to use them, e.g. "
                  "python run_phase.py 2 " + " ".join(phase_specific))
            print()

        for number in CHAIN:
            status = run_phase(number, shared)
            if status != 0:
                print("\nPhase " + str(number) + " exited with status "
                      + str(status) + "; stopping the chain.")
                return status
        print("\nPhases " + str(CHAIN[0]) + "-" + str(CHAIN[-1]) + " completed.")
        print("Phase 9 trains networks and is opt-in: python run_phase.py 9")
        return 0

    try:
        number = int(target)
    except ValueError:
        print("Unrecognised target " + repr(target)
              + ". Expected a phase number, 'all' or 'list'.")
        return 2

    if number not in PHASE_MODULES:
        print("No phase " + str(number) + ". Available: "
              + ", ".join(str(n) for n in PHASE_MODULES))
        return 2

    for required in DEPENDENCIES.get(number, []):
        from brats import config

        directory = config.ARTIFACT_DIR / ("phase" + format(required, "02d"))
        if not directory.is_dir() or not any(directory.glob("*.csv")):
            print("Note: phase " + str(number) + " expects artifacts from phase "
                  + str(required) + ", which has not run yet.")
            print("      Run: python run_phase.py " + str(required))

    return run_phase(number, rest)


if __name__ == "__main__":
    raise SystemExit(main())
