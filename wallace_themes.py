"""Theme presets and saved TUI appearance for Wallace."""

from __future__ import annotations

import json
import os
import re
import stat
import tempfile
import tomllib
from pathlib import Path

from textual.theme import Theme


WALLACE_PRESETS: tuple[Theme, ...] = (
    Theme(
        name="wallace-classic", primary="#7dd3fc", secondary="#c4b5fd",
        success="#86efac", warning="#fcd34d", error="#fb7185",
        foreground="#e0e0e0", background="#121212", surface="#1e1e1e",
    ),
    Theme(
        name="wallace-hacker", primary="#5cff8d", secondary="#31b878",
        success="#8aff8a", warning="#d5eb73", error="#ff6f75",
        foreground="#c8f7d7", background="#050b08", surface="#0d1711",
    ),
    Theme(
        name="wallace-retro", primary="#ffbf69", secondary="#d39054",
        success="#b8c45a", warning="#ffd588", error="#ef866a",
        foreground="#ead1a0", background="#17110a", surface="#23190e",
    ),
    Theme(
        name="wallace-elegant", primary="#c5a46d", secondary="#a89aa7",
        success="#92aa90", warning="#d2ad72", error="#c17d7d",
        foreground="#e9e3d9", background="#17181d", surface="#24242b",
    ),
    Theme(
        name="wallace-future", primary="#52dcff", secondary="#bd7aff",
        success="#62f3c8", warning="#ffd174", error="#ff668f",
        foreground="#e2edff", background="#090f23", surface="#141b33",
    ),
    Theme(
        name="wallace-night-mode", primary="#97b3ae", secondary="#b5a7ba",
        success="#a6bea6", warning="#d0b994", error="#cd9999",
        foreground="#cbd5d1", background="#11191b", surface="#1b2527",
        panel="#202c2e",
    ),
    Theme(
        name="wallace-red-light", primary="#d00000", secondary="#a80000",
        success="#b80000", warning="#e00000", error="#ff0000",
        accent="#c00000", foreground="#d80000", background="#050000",
        surface="#180000", panel="#200000", luminosity_spread=0.0,
        variables={
            "text": "#d80000",
            "text-primary": "#d00000",
            "text-secondary": "#a80000",
            "text-success": "#b80000",
            "text-warning": "#e00000",
            "text-error": "#ff0000",
            "text-accent": "#c00000",
            "text-muted": "#890000",
            "text-disabled": "#650000",
            "button-foreground": "#d80000",
            "button-color-foreground": "#100000",
            "footer-key-foreground": "#ff0000",
            "input-selection-background": "#610000",
            "input-selection-foreground": "#ff0000",
            "screen-selection-background": "#610000",
            "screen-selection-foreground": "#ff0000",
        },
    ),
)

DEFAULT_THEME = "wallace-classic"
_HEX_COLOR = re.compile(r"#[0-9a-fA-F]{6}\Z")


def _color(values: dict, key: str, fallback: str | None = None) -> str | None:
    value = values.get(key)
    if isinstance(value, str) and _HEX_COLOR.fullmatch(value):
        return value
    return fallback


def discover_omarchy_themes(
    system_dir: Path = Path("/usr/share/omarchy/themes"),
    user_dir: Path | None = None,
) -> list[Theme]:
    """Read installed Omarchy colors without changing the system theme.

    A user theme of the same name overlays stock colors, as Omarchy does.
    Other operating systems simply return no Omarchy entries.
    """
    if user_dir is None:
        user_dir = Path.home() / ".config" / "omarchy" / "themes"
    names: set[str] = set()
    for directory in (system_dir, user_dir):
        if directory.is_dir():
            try:
                names.update(path.name for path in directory.iterdir() if path.is_dir())
            except OSError:
                continue

    themes: list[Theme] = []
    for name in sorted(names):
        values: dict = {}
        for directory in (system_dir, user_dir):
            path = directory / name / "colors.toml"
            try:
                with path.open("rb") as source:
                    values.update(tomllib.load(source))
            except (OSError, tomllib.TOMLDecodeError):
                continue
        primary = _color(values, "accent")
        background = _color(values, "background")
        foreground = _color(values, "foreground")
        if not (primary and background and foreground):
            continue
        themes.append(
            Theme(
                name=f"omarchy-{name}",
                primary=primary,
                secondary=_color(values, "blue", primary),
                success=_color(values, "green", primary),
                warning=_color(values, "yellow", primary),
                error=_color(values, "red", primary),
                foreground=foreground,
                background=background,
                surface=_color(values, "lighter_background", background),
                dark=values.get("mode") != "light",
            )
        )
    return themes


def rich_palette(theme: Theme) -> dict[str, str]:
    """Resolve Textual theme colors to styles used by Rich log entries."""
    colors = theme.to_color_system().generate()

    def rich_color(value: str) -> str:
        return value.removeprefix("ansi_")

    palette = {
        "primary": rich_color(colors["primary"]),
        "secondary": rich_color(colors["secondary"]),
        "success": rich_color(colors["success"]),
        "warning": rich_color(colors["warning"]),
        "error": rich_color(colors["error"]),
        "user": rich_color(colors["primary"]),
        "assistant": rich_color(colors["secondary"]),
        "muted": "dim",
    }
    if theme.name == DEFAULT_THEME:
        palette["user"] = "#93c5fd"
        palette["muted"] = "#94a3b8"
    elif theme.name == "wallace-red-light":
        palette["muted"] = "#890000"
    return palette


def load_theme_choice(project_root: Path) -> str:
    try:
        data = json.loads((project_root / ".wallace" / "ui.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return DEFAULT_THEME
    choice = data.get("theme") if isinstance(data, dict) else None
    return choice if isinstance(choice, str) and choice else DEFAULT_THEME


def save_theme_choice(project_root: Path, theme_name: str) -> None:
    path = project_root / ".wallace" / "ui.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=".ui.", suffix=".tmp", delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump({"theme": theme_name}, temporary)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
        try:
            os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
        except OSError:
            pass
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
