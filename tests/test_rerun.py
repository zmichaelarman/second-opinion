from second_opinion.rerun import verdict


def test_rerun_verdicts():
    assert verdict(0, 30, 30, 30)[0] == "fixed"
    assert verdict(20, 30, 30, 30)[0] == "improved"
    assert verdict(29, 30, 30, 30)[0] == "still failing"
    assert verdict(0, 30, 1, 30)[0] == "still failing"  # cannot tell 0/30 from 1/30
