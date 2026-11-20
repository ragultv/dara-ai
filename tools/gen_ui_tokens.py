"""Generates the Dara design tokens: docs/ui/tokens.css (CSS variables, light + dark) and docs/ui/contrast-report.md (WCAG checks of every pair).

Usage: python -I tools/gen_ui_tokens.py

Brand: PURPLE is primary, WHITE is secondary, BLACK is tertiary.
  * purple scale   - generated in OKLCH (hue 295), gamut-mapped to sRGB
  * white / paper  - white and very light purple-tinted neutrals (surfaces in the light theme, text on dark)
  * black / ink    - near-black and purple-tinted dark neutrals (text in the light theme, surfaces in the dark theme)
Semantic tokens (--color-*) are what the app uses; the raw scales are for building them. The script fails if a required pair misses its WCAG target.
"""
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HUE = 295.0


# ---- color math -----------------------------------------------------------------------------------------------------------------------------------------
def oklch_to_linear(L, C, h):
    a, b = C * math.cos(math.radians(h)), C * math.sin(math.radians(h))
    l_, m_, s_ = L + 0.3963377774 * a + 0.2158037573 * b, L - 0.1055613458 * a - 0.0638541728 * b, L - 0.0894841775 * a - 1.2914855480 * b
    l, m, s = l_ ** 3, m_ ** 3, s_ ** 3
    return (4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
            -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
            -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s)


def in_gamut(rgb):
    return all(-1e-4 <= c <= 1 + 1e-4 for c in rgb)


def encode(c):
    c = min(1.0, max(0.0, c))
    return 12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055


def oklch_hex(L, C, h):
    """Reduces the chroma until the color is inside sRGB (keeps lightness and hue)."""
    while C > 0 and not in_gamut(oklch_to_linear(L, C, h)):
        C -= 0.002
    r, g, b = (encode(c) for c in oklch_to_linear(L, max(C, 0), h))
    return "#%02X%02X%02X" % (round(r * 255), round(g * 255), round(b * 255))


def lin(c8):
    c = c8 / 255
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def luminance(hexv):
    r, g, b = (int(hexv[i:i + 2], 16) for i in (1, 3, 5))
    return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)


def contrast(a, b):
    la, lb = luminance(a), luminance(b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


# ---- scales ---------------------------------------------------------------------------------------------------------------------------------------------
PURPLE = {50: (0.975, 0.012), 100: (0.948, 0.030), 200: (0.905, 0.060), 300: (0.835, 0.110), 400: (0.735, 0.170), 500: (0.640, 0.225), 600: (0.550, 0.240),
          700: (0.470, 0.215), 800: (0.390, 0.175), 900: (0.310, 0.125), 950: (0.225, 0.085)}
INK = {50: (0.975, 0.004), 100: (0.950, 0.006), 200: (0.912, 0.009), 300: (0.860, 0.012), 400: (0.720, 0.016), 500: (0.585, 0.020), 600: (0.480, 0.022),
       700: (0.385, 0.024), 800: (0.300, 0.024), 900: (0.225, 0.022), 950: (0.165, 0.018)}
# Royal violet (primary #7C3AED family). Explicit hexes: the brand palette, not a generated ramp.
scale_purple = {50: "#F5F0FF", 100: "#EBDDFF", 200: "#D6BBFF", 300: "#B994FF", 400: "#9A6BFA", 500: "#7C3AED", 600: "#6D28D9", 700: "#5B21B6", 800: "#4C1D95",
                900: "#2E1065", 950: "#1E0A4A"}
scale_ink = {k: oklch_hex(L, C, HUE) for k, (L, C) in INK.items()}
WHITE, BLACK, BLACK_UI = "#FFFFFF", "#000000", "#0F0F0F"   # BLACK: logo / print; BLACK_UI: the dark theme's page background (pure black glares on LCDs; this one is a purple-tinted near-black)

STATUS = {   # functional colors: not brand, harmonised in lightness with the purple scale
    "success": {"light": "#15803D", "dark": "#4ADE80", "tint_light": "#ECFDF3", "tint_dark": "#0E2A1A"},
    "warning": {"light": "#A15C07", "dark": "#FBBF24", "tint_light": "#FFF7E6", "tint_dark": "#2E2108"},
    "danger": {"light": "#C81E1E", "dark": "#FF8A8A", "tint_light": "#FEF0F0", "tint_dark": "#3A1212"},
}

# Purple is reserved for BUTTONS (--color-primary*): everything else (surfaces, text, borders, focus, links, selected states) is neutral grey.
LIGHT = {
    "bg": "#FAFAFA", "bg-subtle": "#F4F4F5", "surface": WHITE, "surface-raised": "#EBEBEF", "surface-sunken": "#F4F4F5",
    "border": "#E4E4E7", "border-strong": "#85858F",
    "text": "#18181B", "text-muted": "#52525B", "text-subtle": "#71717A",
    "primary": scale_purple[700], "primary-hover": scale_purple[800], "primary-active": scale_purple[900], "on-primary": WHITE,
    "primary-subtle": "#EBEBEF", "primary-subtle-hover": "#E4E4E7", "on-primary-subtle": "#18181B",
    "link": "#18181B", "focus": "#18181B",
    "success": STATUS["success"]["light"], "success-subtle": STATUS["success"]["tint_light"],
    "warning": STATUS["warning"]["light"], "warning-subtle": STATUS["warning"]["tint_light"],
    "danger": STATUS["danger"]["light"], "danger-subtle": STATUS["danger"]["tint_light"],
    "code-bg": "#F4F4F5", "overlay": "rgb(9 9 11 / 0.45)",
}
DARK = {   # neutral dark greys with visible layers: page < sidebar < card < raised
    "bg": "#0F0F0F", "bg-subtle": "#171717", "surface": "#1E1E1E", "surface-raised": "#242424", "surface-sunken": "#141414",
    "border": "#2E2E2E", "border-strong": "#7A7A7A",
    "text": "#E5E5E5", "text-muted": "#A3A3A3", "text-subtle": "#8A8A8A",
    "primary": scale_purple[600], "primary-hover": scale_purple[500], "primary-active": scale_purple[700], "on-primary": WHITE,
    "primary-subtle": "#2A2A2A", "primary-subtle-hover": "#333333", "on-primary-subtle": "#E5E5E5",
    "link": "#E5E5E5", "focus": "#E5E5E5",
    "success": STATUS["success"]["dark"], "success-subtle": STATUS["success"]["tint_dark"],
    "warning": STATUS["warning"]["dark"], "warning-subtle": STATUS["warning"]["tint_dark"],
    "danger": STATUS["danger"]["dark"], "danger-subtle": STATUS["danger"]["tint_dark"],
    "code-bg": "#161616", "overlay": "rgb(0 0 0 / 0.6)",
}

# (foreground token, background token, minimum ratio, what it is)
CHECKS = [
    ("text", "bg", 7.0, "body text on the page"), ("text", "surface", 7.0, "body text on cards"), ("text", "surface-raised", 4.5, "text on raised surfaces"),
    ("text-muted", "bg", 4.5, "secondary text on the page"), ("text-muted", "surface", 4.5, "secondary text on cards"), ("text-muted", "surface-raised", 4.5, "secondary text on raised surfaces"),
    ("text-subtle", "bg", 3.0, "placeholder / disabled-adjacent text (large or decorative only)"),
    ("on-primary", "primary", 4.5, "text on a primary button"), ("on-primary", "primary-hover", 4.5, "text on a hovered primary button"), ("on-primary", "primary-active", 4.5, "text on a pressed primary button"),
    ("link", "bg", 4.5, "link text on the page"), ("link", "surface", 4.5, "link text on cards"),
    ("on-primary-subtle", "primary-subtle", 4.5, "text on a subtle purple chip / selected row"),
    ("focus", "bg", 3.0, "focus ring on the page"), ("focus", "surface", 3.0, "focus ring on cards"),
    ("border-strong", "bg", 3.0, "input borders (UI component boundary)"),
    ("success", "bg", 4.5, "success text"), ("warning", "bg", 4.5, "warning text"), ("danger", "bg", 4.5, "error text"),
    ("success", "success-subtle", 4.5, "success text on its tint"), ("warning", "warning-subtle", 4.5, "warning text on its tint"), ("danger", "danger-subtle", 4.5, "error text on its tint"),
]


def run_checks():
    rows, failures = [], []
    for theme, tokens in (("light", LIGHT), ("dark", DARK)):
        for fg, bg, need, what in CHECKS:
            r = contrast(tokens[fg], tokens[bg])
            ok = r >= need
            rows.append((theme, fg, bg, tokens[fg], tokens[bg], r, need, ok, what))
            if not ok: failures.append((theme, fg, bg, r, need))
    return rows, failures


def write_tokens(path):
    out = ["/* Dara design tokens. GENERATED by tools/gen_ui_tokens.py -- do not edit by hand; change the script and rerun it.",
           "   Brand: purple = primary, white = secondary, black = tertiary. Use the semantic --color-* tokens in the app; the raw scales are for building them. */", "",
           ":root {", "  /* purple (primary) */"]
    out += [f"  --dara-purple-{k}: {v};" for k, v in scale_purple.items()]
    out += ["  /* white (secondary) and black (tertiary) */", f"  --dara-white: {WHITE};", f"  --dara-black: {BLACK};          /* logo, print */",
            f"  --dara-black-ui: {BLACK_UI};       /* the dark theme's page background */", "  /* ink: purple-tinted neutrals between white and black */"]
    out += [f"  --dara-ink-{k}: {v};" for k, v in scale_ink.items()]
    out += ["", "  /* typography */",
            "  --font-ui: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;",
            "  --font-display: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;",
            "  --font-mono: 'Cascadia Mono', 'SF Mono', Consolas, 'Liberation Mono', Menlo, Courier, monospace;",
            "  --text-xs: 0.75rem; --text-sm: 0.8125rem; --text-base: 0.9375rem; --text-lg: 1.0625rem; --text-xl: 1.25rem; --text-2xl: 1.5rem; --text-3xl: 1.875rem;",
            "  --leading-tight: 1.25; --leading-normal: 1.5; --leading-relaxed: 1.65;",
            "  /* space: 4px grid */",
            "  --space-1: 0.25rem; --space-2: 0.5rem; --space-3: 0.75rem; --space-4: 1rem; --space-5: 1.25rem; --space-6: 1.5rem; --space-8: 2rem; --space-10: 2.5rem; --space-12: 3rem; --space-16: 4rem;",
            "  /* shape */", "  --radius-sm: 6px; --radius-md: 10px; --radius-lg: 14px; --radius-xl: 20px; --radius-full: 9999px;",
            "  /* motion: short, and off when the user asks (see prefers-reduced-motion below) */",
            "  --ease: cubic-bezier(0.2, 0, 0, 1); --duration-fast: 120ms; --duration-base: 200ms; --duration-slow: 320ms;",
            "  /* layers */", "  --z-sticky: 10; --z-drawer: 30; --z-modal: 40; --z-toast: 50;", "}", ""]

    def block(selector, tokens, shadow):
        lines = [f"{selector} {{", "  color-scheme: " + ("dark" if tokens is DARK else "light") + ";"]
        lines += [f"  --color-{k}: {v};" for k, v in tokens.items()]
        lines += [f"  --shadow-sm: {shadow[0]};", f"  --shadow-md: {shadow[1]};", f"  --shadow-lg: {shadow[2]};", "}", ""]
        return lines
    light_shadow = ("0 1px 2px rgb(9 9 11 / 0.06)", "0 8px 24px rgb(9 9 11 / 0.10)", "0 12px 32px rgb(9 9 11 / 0.16)")
    dark_shadow = ("0 1px 2px rgb(8 3 18 / 0.5)", "0 4px 12px rgb(8 3 18 / 0.55)", "0 12px 32px rgb(8 3 18 / 0.65)")
    out += ["/* light (default) */"] + block(":root,\n:root[data-theme='light']", LIGHT, light_shadow)
    out += ["/* dark: explicit choice, or the system setting when the user has not chosen */"] + block(":root[data-theme='dark']", DARK, dark_shadow)
    out += ["@media (prefers-color-scheme: dark) {"]
    out += ["  " + l if l else l for l in block(":root:not([data-theme='light'])", DARK, dark_shadow)]
    out += ["}", "", "@media (prefers-reduced-motion: reduce) {", "  :root { --duration-fast: 0ms; --duration-base: 0ms; --duration-slow: 0ms; }", "}", ""]
    open(path, "w", newline="\n").write("\n".join(out))


def write_report(path, rows):
    out = ["# Contrast report", "", "Generated by `tools/gen_ui_tokens.py` (WCAG 2.2 contrast ratio: (L1 + 0.05) / (L2 + 0.05)). Every row must pass; the script fails otherwise.", "",
           "Targets: body text 7:1 (AAA), other text 4.5:1 (AA), icons / borders / focus rings / large decorative text 3:1.", ""]
    for theme in ("light", "dark"):
        out += [f"## {theme.capitalize()} theme", "", "| Foreground | Background | Colors | Ratio | Needs | What it is |", "| --- | --- | --- | --- | --- | --- |"]
        for t, fg, bg, fgv, bgv, r, need, ok, what in rows:
            if t == theme: out.append(f"| `{fg}` | `{bg}` | {fgv} on {bgv} | **{r:.2f}** {'ok' if ok else 'FAIL'} | {need:g} | {what} |")
        out.append("")
    out += ["## Scales", "", "| Step | Purple | Ink |", "| --- | --- | --- |"] + [f"| {k} | {scale_purple[k]} | {scale_ink[k]} |" for k in scale_purple]
    out += ["", f"White {WHITE}, black {BLACK}, dark-theme page background {BLACK_UI}.", ""]
    open(path, "w", newline="\n").write("\n".join(out))


if __name__ == "__main__":
    rows, failures = run_checks()
    os.makedirs(os.path.join(ROOT, "docs", "ui"), exist_ok=True)
    write_tokens(os.path.join(ROOT, "docs", "ui", "tokens.css"))
    write_report(os.path.join(ROOT, "docs", "ui", "contrast-report.md"), rows)
    print(f"purple: {scale_purple}")
    print(f"ink:    {scale_ink}")
    for theme, fg, bg, r, need in failures: print(f"FAIL {theme}: {fg} on {bg} = {r:.2f} (needs {need})")
    print(f"{len(rows) - len(failures)} of {len(rows)} pairs pass")
    sys.exit(1 if failures else 0)
