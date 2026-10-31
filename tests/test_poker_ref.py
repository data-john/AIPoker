from aipoker.poker_ref import (
    RFI_WIDTH_PCT,
    combos,
    hand_classes,
    percentile_table,
    reference_action,
    spearman,
    strength_order,
)


def test_169_classes_and_1326_combos():
    classes = hand_classes()
    assert len(classes) == len(set(classes)) == 169
    assert sum(combos(c) for c in classes) == 1326


def test_strength_ordering_sanity():
    order = strength_order()
    assert order[0] == "AA"
    assert order.index("AKs") < order.index("AKo") < order.index("72o")
    assert order.index("KK") < order.index("QQ") < order.index("JJ")
    assert order[-1] in {"72o", "32o", "42o", "52o", "62o"}
    assert abs(percentile_table()[order[-1]] - 100.0) < 1e-9


def test_reference_widths_increase_toward_the_button():
    opens = {
        pos: sum(combos(c) for c in hand_classes() if reference_action(c, pos) == "raise")
        for pos in RFI_WIDTH_PCT
    }
    assert opens["UTG"] < opens["HJ"] < opens["CO"] < opens["BTN"]
    assert reference_action("AA", "UTG") == "raise" and reference_action("72o", "BTN") == "fold"


def test_spearman():
    assert abs(spearman([1, 2, 3, 4], [10, 20, 30, 40]) - 1) < 1e-9
    assert abs(spearman([1, 2, 3, 4], [4, 3, 2, 1]) + 1) < 1e-9
