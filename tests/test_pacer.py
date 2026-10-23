import pytest

from second_opinion.pacer import Pacer, parse_duration


class FakeTime:
    def __init__(self):
        self.t = 0.0
        self.slept: list[float] = []

    def clock(self):
        return self.t

    async def sleep(self, seconds):
        self.slept.append(seconds)
        self.t += seconds


def pacer(rpm=60):
    ft = FakeTime()
    return Pacer(rpm, clock=ft.clock, sleep=ft.sleep), ft


@pytest.mark.parametrize(
    "value,seconds",
    [("1.5", 1.5), ("20ms", 0.02), ("6m0s", 360.0), ("1h2m3s", 3723.0), ("", None), (None, None), ("soon", None)],
)
def test_parse_duration(value, seconds):
    assert parse_duration(value) == seconds


async def test_requests_are_spaced_at_the_rpm():
    p, ft = pacer(60)
    for _ in range(3):
        await p.acquire()
    assert ft.t == pytest.approx(2.0)


async def test_rate_is_capped_by_80_percent_of_the_header_limit():
    p, ft = pacer(60)
    p.observe({"x-ratelimit-limit-requests": "30"})
    assert p.rate == pytest.approx(24.0)
    await p.acquire()
    await p.acquire()
    assert ft.t == pytest.approx(2.5)


async def test_over_limit_halves_the_rate_for_120_seconds():
    p, ft = pacer(40)
    p.observe({"x-ratelimit-over-limit": "yes"})
    assert p.rate == pytest.approx(20.0)
    ft.t += 121
    assert p.rate == pytest.approx(40.0)


async def test_low_remaining_tokens_waits_for_reset_once():
    p, ft = pacer(60)
    p.observe({"x-ratelimit-remaining-tokens": "1000", "x-ratelimit-reset-tokens": "7s"})
    await p.acquire(est_tokens=2400)
    assert ft.t == pytest.approx(7.0)
    await p.acquire(est_tokens=2400)  # no fresh header: does not wait again
    assert ft.t == pytest.approx(8.0)


async def test_pause_after_429_holds_everyone():
    p, ft = pacer(60)
    p.pause(30)
    await p.acquire()
    assert ft.t == pytest.approx(30.0)
