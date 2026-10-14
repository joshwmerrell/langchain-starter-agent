import unittest
from datetime import datetime, timedelta, timezone

from rich.cells import cell_len

from wallace_sky import moon_phase, sky_icons


class SkyIndicatorTests(unittest.TestCase):
    def test_reference_and_primary_2026_phases(self):
        # US Naval Observatory primary phase times, rounded to the minute.
        expected = (
            ((2026, 10, 10, 15, 50), "○", "New Moon"),
            ((2026, 10, 18, 16, 12), "◐", "First Quarter"),
            ((2026, 10, 26, 4, 12), "●", "Full Moon"),
            ((2026, 11, 1, 20, 28), "◑", "Last Quarter"),
        )
        for parts, icon, name in expected:
            with self.subTest(phase=name):
                at = datetime(*parts, tzinfo=timezone.utc)
                self.assertEqual(moon_phase(at), (icon, name))

    def test_moon_phase_uses_the_instant_not_the_local_timezone(self):
        utc_time = datetime(2026, 10, 14, 12, tzinfo=timezone.utc)
        local_time = utc_time.astimezone(timezone(timedelta(hours=-6)))
        self.assertEqual(moon_phase(utc_time), moon_phase(local_time))
        self.assertEqual(moon_phase(utc_time)[0], "☽")

    def test_sky_indicators_change_at_local_day_night_boundaries(self):
        local = timezone(timedelta(hours=-6))
        for hour, minute, expected in (
            (5, 59, "☽"), (6, 0, "☀"),
            (17, 59, "☀"), (18, 0, "☽"),
        ):
            with self.subTest(hour=hour, minute=minute):
                at = datetime(2026, 10, 14, hour, minute, tzinfo=local)
                icon, tooltip = sky_icons(at)
                self.assertEqual(icon, expected)
                self.assertEqual(cell_len(icon), 1)
                self.assertIn("approximate", tooltip)

    def test_emoji_mode_keeps_one_icon_of_equal_terminal_width(self):
        local = timezone(timedelta(hours=-6))
        day = datetime(2026, 10, 14, 12, tzinfo=local)
        night = day.replace(hour=22)
        day_icon, _ = sky_icons(day, emoji=True)
        night_icon, _ = sky_icons(night, emoji=True)
        self.assertEqual(day_icon, "☀️")
        self.assertEqual(night_icon, "🌒")
        self.assertEqual(cell_len(day_icon), cell_len(night_icon))

    def test_all_default_moon_phases_are_one_cell_text_icons(self):
        new_moon = datetime(2026, 10, 10, 15, 50, tzinfo=timezone.utc)
        for day in range(30):
            with self.subTest(day=day):
                icon, _ = moon_phase(new_moon + timedelta(days=day))
                self.assertEqual(cell_len(icon), 1)

    def test_optional_future_weather_icon_replaces_the_sky_icon(self):
        at = datetime(2026, 10, 14, 12, tzinfo=timezone.utc)
        icon, _ = sky_icons(at, weather_icon="☁")
        self.assertEqual(icon, "☁")

    def test_naive_time_is_rejected(self):
        with self.assertRaises(ValueError):
            moon_phase(datetime(2026, 10, 14, 12))
