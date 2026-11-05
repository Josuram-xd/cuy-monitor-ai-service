import pytest

from dev.inject_test_data import build_parser


def test_the_injector_defaults_to_a_lethargic_guinea_pig_every_minute():
    args = build_parser().parse_args(["--api-key", "k", "behavior"])

    assert (args.scenario, args.windows, args.interval) == ("lethargic", 10, 60.0)


def test_the_injector_rejects_unknown_scenarios_and_zero_windows():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["behavior", "--scenario", "sleepy"])
    with pytest.raises(SystemExit):
        build_parser().parse_args(["behavior", "--windows", "0"])


def test_the_injector_sends_squeals_by_default():
    args = build_parser().parse_args(["audio", "--count", "2"])

    assert (args.kind, args.count) == ("squeal", 2)
