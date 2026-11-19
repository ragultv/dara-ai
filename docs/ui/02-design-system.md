# Design system

Source of truth for color and tokens: [tokens.css](tokens.css) (generated; `python -I tools/gen_ui_tokens.py` regenerates it and fails if any pair misses WCAG). Contrast numbers: [contrast-report.md](contrast-report.md).

## Brand roles

| Role | Color | Used for |
| --- | --- | --- |
| Primary | **Purple** (hue 295; `--dara-purple-50..950`) | Buttons, selected states, links, focus ring, charts, the mark |
| Secondary | **White** (and very light purple-tinted whites) | Light theme surfaces; text on purple and on the dark theme |
| Tertiary | **Black** (and purple-tinted near-blacks, `--dara-ink-*`) | Text in the light theme; the dark theme's surfaces; the logo |

Purple is the only accent. Status colors (green / amber / red) are functional, never decorative.

## Themes

- Light (default for the first run) and Dark. Setting: System / Light / Dark; the `data-theme` attribute on `<html>` overrides the OS (`light` or `dark`); with no attribute the OS decides.
- Dark background is `#0B0A10` (`--dara-black-ui`), not pure black (pure black glares on laptop LCDs and smears on OLED). Pure `#000000` is only for the logo and print.
- Always use the semantic tokens:

| Token | Meaning |
| --- | --- |
| `--color-bg`, `--color-bg-subtle` | Page, and a slightly tinted band (sidebars) |
| `--color-surface`, `--color-surface-raised`, `--color-surface-sunken` | Cards, popovers/menus, wells (inputs, code) |
| `--color-border`, `--color-border-strong` | Dividers; input and control outlines (3:1 against the page) |
| `--color-text`, `--color-text-muted`, `--color-text-subtle` | Body (7:1), secondary (4.5:1), hints (3:1, large or decorative only) |
| `--color-primary`, `-hover`, `-active`, `--color-on-primary` | Primary buttons and filled states, text on them |
| `--color-primary-subtle`, `-subtle-hover`, `--color-on-primary-subtle` | Selected rows, chips, the assistant's accent areas |
| `--color-link`, `--color-focus` | Links; focus ring |
| `--color-success/warning/danger` (+ `-subtle`) | Status text and its tinted background |
| `--color-code-bg`, `--color-overlay` | Code blocks; modal scrim |

Rules: no hex or `rgb()` in components; no `opacity` to fake a color (use a token); do not put `--color-text-subtle` on anything a user must read.

## Typography

System fonts only (no font downloads, no extra memory): `--font-ui` (Segoe UI Variable), `--font-mono` (Cascadia Mono / Consolas). Scale: `--text-xs` 12, `sm` 13, `base` 15, `lg` 17, `xl` 20, `2xl` 24, `3xl` 30 px. Body is `--text-base` / `--leading-normal`. Chat message text uses `--leading-relaxed`. Numbers that change live (tok/s, VRAM) use `font-variant-numeric: tabular-nums` so they do not jitter.

## Space, shape, motion

4 px grid via `--space-*`. Radius: controls `--radius-md` (10), cards `--radius-lg` (14), chips `--radius-full`. Shadows only on floating layers (menus, modals), not on cards; use borders. Motion: 120 to 320 ms with `--ease`; nothing loops except the generating indicator. `prefers-reduced-motion` zeroes the durations (already in tokens.css); the generating indicator must then be static text.

## Layout

Min window 900 x 600. Left rail 64 px (icons) or 232 px (icons + labels). Content max width: chat 760 px centered; settings/options 960 px. Use CSS grid/flex, no fixed pixel widths for text. Must work at 125 % and 150 % Windows scaling and with the window at the minimum size.

## Components (build once in `app/src/lib/ui/`, reuse everywhere)

Button (primary, secondary, ghost, danger; sizes sm/md), IconButton (needs `aria-label`), Input, Textarea (auto-grow), Select, SegmentedControl, Toggle, Slider, Checkbox, Card, Badge (neutral, purple, success, warning, danger), Tooltip, Popover, Menu, Dialog (focus trap), Toast, ProgressBar (determinate + indeterminate), Skeleton, Tabs, EmptyState, ErrorState, Meter (context/VRAM), Sparkline, CodeBlock, Markdown.

Primary button: filled `--color-primary` with `--color-on-primary` text. Secondary button: `--color-surface` with `--color-border-strong` outline and `--color-text`. One primary button per view. Destructive: `--color-danger`, always confirmed in a Dialog that names what will be lost.

Quality badges (options screen): High = success, Medium = purple, Experimental = warning, Lower = neutral. Each carries its text label and an icon; color is never the only signal.

## Chat look

User messages: right-aligned bubble on `--color-primary-subtle` with `--color-on-primary-subtle`. Assistant messages: no bubble, plain text on the page, with a small purple Dara mark. Code blocks: `--color-code-bg`, mono, copy button. The context meter and per-message stats are muted and small. The "thinking/generating" state is a pulsing purple dot (static under reduced motion).

## Identity

Own mark and wordmark ("Dara", lowercase-friendly geometric sans), purple on white or white on purple; a single-color black version for print. Do not reuse any other product's mascot, wordmark, splash or phrases. Empty states use simple geometric illustrations in `--dara-purple-*`, not stock art.

## Accessibility (required, tested)

- WCAG 2.2 AA minimum; text contrast is verified in the contrast report; keep it that way when adding tokens.
- Everything works by keyboard: visible focus ring (`outline: 2px solid var(--color-focus); outline-offset: 2px`), logical tab order, Esc closes popovers and dialogs, focus returns to the trigger.
- Screen readers: semantic HTML first (`button`, `nav`, `main`, `dialog`), ARIA only to fill gaps; streaming replies live in an `aria-live="polite"` region that updates by sentence, not by token; progress uses `role="progressbar"` with values.
- Hit targets at least 32 x 32 px (40 for primary actions). Respect Windows high contrast (`forced-colors`): borders and focus must remain visible.
- Text can be scaled to 200 % without loss.
