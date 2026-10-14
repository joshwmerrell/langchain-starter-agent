"""Small offline sky indicators for Wallace's TUI header.

The phase is an approximation based on a mean synodic month, not an ephemeris.
The reference new moon is 2026-10-10 15:50 UTC (US Naval Observatory).
"""

from __future__ import annotations

from datetime import datetime, timezone


# https://aa.usno.navy.mil/calculated/moon/phases?date=2026-01-01&format=p&nump=50
REFERENCE_NEW_MOON_UTC = datetime(2026, 10, 10, 15, 50, tzinfo=timezone.utc)
# https://eclipse.gsfc.nasa.gov/SEsaros/SEpanorama.html
MEAN_SYNODIC_DAYS = 29.530589
MOON_PHASES = (
    # text symbol, emoji, accessible name; text symbols occupy one terminal cell
    ("○", "🌑", "New Moon"),
    ("☽", "🌒", "Waxing Crescent"),
    ("◐", "🌓", "First Quarter"),
    ("◕", "🌔", "Waxing Gibbous"),
    ("●", "🌕", "Full Moon"),
    ("◕", "🌖", "Waning Gibbous"),
    ("◑", "🌗", "Last Quarter"),
    ("☾", "🌘", "Waning Crescent"),
)


def moon_phase(at: datetime, *, emoji: bool = False) -> tuple[str, str]:
    """Return the closest approximate moon phase as a text or emoji icon."""
    if at.tzinfo is None or at.utcoffset() is None:
        raise ValueError("moon_phase requires a timezone-aware datetime")
    elapsed_days = (at.astimezone(timezone.utc) - REFERENCE_NEW_MOON_UTC).total_seconds() / 86_400
    phase_fraction = (elapsed_days / MEAN_SYNODIC_DAYS) % 1.0
    index = int(phase_fraction * len(MOON_PHASES) + 0.5) % len(MOON_PHASES)
    text_icon, emoji_icon, name = MOON_PHASES[index]
    return (emoji_icon if emoji else text_icon), name


def sky_icons(
    at: datetime, *, emoji: bool = False, weather_icon: str | None = None,
) -> tuple[str, str]:
    """Return one header icon and an explanatory tooltip.

    Day/night uses local 06:00–18:00 until a future location-aware weather
    source provides actual sunrise and sunset. ``weather_icon`` can replace
    the icon later; Wallace does not fetch weather here.
    """
    phase_icon, phase_name = moon_phase(at, emoji=emoji)
    daytime = 6 <= at.hour < 18
    icon = ("☀️" if emoji else "☀") if daytime else phase_icon
    if weather_icon is not None:
        icon = weather_icon
    period = "Daytime" if daytime else "Nighttime"
    tooltip = (
        f"{period} (approximate local 06:00–18:00); "
        f"{phase_name} (approximate moon phase)."
    )
    return icon, tooltip
