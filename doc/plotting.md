# Figure Creation Guide

## When to use this file

Read this guide before creating or revising any publication, manuscript,
presentation, ERP, topographic, or statistical figure. It defines the required
Nature Neuroscience-style conventions for this project.

## Nature Neuroscience-style appearance

- Use minimal, publication-quality scientific graphics on a white background.
- Do not use decorative elements, shadows, gradients, 3D effects, or dense
  gridlines.
- Keep visual styling, condition order, and color mappings consistent across
  manuscript figures.
- Prioritize readability and the data over decoration.
- Do not use large titles inside plots; scientific interpretation belongs in
  the figure legend.

## Figure size, typography, and export

Design figures at their intended publication size:

- Single-column width: approximately 90 mm (3.5 in).
- Double-column width: approximately 180 mm (7.1 in); never exceed this width.
- Two horizontal panels: approximately 7.0 x 3.0--3.5 in.
- Full-width multi-panel figures: approximately 7.0 x 5--7 in.

Use Times New Roman when available, with Times or DejaVu Serif as fallbacks.
Use thin axes and ticks (approximately 0.5--0.8 pt), outward-facing ticks, and
left/bottom spines only unless additional spines are scientifically necessary.

### Typography profiles: paper versus poster

When a figure request identifies its destination, use the corresponding
profile below. If the destination is not stated, ask whether the figure is for
a paper/publication or a poster before selecting typography. Use Times New
Roman for both profiles.

| Setting | Paper / publication profile | Poster profile |
|---|---:|---:|
| Title | Avoid a large in-plot title; use the manuscript figure legend instead | 15 pt, bold when a title is requested |
| Axis labels | approximately 7 pt | 12 pt |
| Tick labels | 6--7 pt | 10 pt |
| Legend text | 6--7 pt | 10 pt |
| Significance annotations | match tick-label size unless a figure-specific need requires otherwise | 10 pt |
| Panel labels | 8--9 pt, bold lowercase | 12 pt, bold lowercase |

The poster profile records the settings used for
`behavioral_rt_change_barplot`: 15 pt title, 12 pt axis labels, and 10 pt
ticks, legend, and significance labels. It is intended for the final displayed
poster figure size; adjust the physical canvas dimensions to the poster layout
while retaining this visual hierarchy unless the user specifies otherwise.

Export both a vector PDF (or SVG) and a PNG preview at at least 300 dpi. Use
`bbox_inches="tight"`, then verify that no legend, panel label, or other
important element was clipped. Do not rasterize text, axes, labels, scale bars,
or other vector elements unless technically necessary.

## Colors, legends, and multi-panel layout

- Use color sparingly; it must be colorblind-accessible and distinguishable in
  print. Never use red-versus-green comparisons or rainbow/jet colormaps.
- Use muted scientific colors and a fixed condition-to-color mapping across all
  figures. Use black or gray when color is not necessary.
- Use perceptually uniform sequential colormaps (for example, viridis, magma,
  or cividis) for continuous nonnegative maps.
- Use compact, frameless legends. Prefer a single shared legend in multi-panel
  figures; do not repeat legends unnecessarily or place them over data.
- Use lowercase bold panel labels (`a`, `b`, `c`, ...) at a consistent
  upper-left position, outside or just inside the plotting region without
  obscuring data.
- Align panel margins, axes, fonts, and formatting intentionally. Use shared
  labels where they reduce redundant text.

## Project-specific EEG figure rules

### Comparable sessions, conditions, and electrodes

- Never choose axis limits independently for panels that are directly compared.
- Never choose topoplot color limits independently for comparable maps.
- Determine shared limits across the complete comparison before plotting and
  report those limits in the figure or its legend when relevant.
- Maintain the same electrode-condition color mapping across every figure.

### ERP plots

- Show stimulus onset at t = 0 with a thin, subtle vertical reference line.
- Shade prespecified analysis windows subtly.
- Do not smooth data unless explicitly requested.
- Do not visually emphasize a time window selected after inspecting the data.
- Use thin-to-medium traces (approximately 1--1.5 pt), light SEM or confidence
  interval bands when appropriate, no dense background grid, and identical
  y-axis scales for comparable panels.
- Plot negative voltage upward only when explicitly requested; otherwise use
  the established manuscript convention and label it clearly.

### Topoplots

- Use identical electrode layouts and head geometry across comparable maps.
- For r², Fisher score, and other nonnegative discriminability metrics, use a
  sequential colormap with one shared range across sessions or conditions.
- For signed amplitudes and contra-minus-ipsilateral quantities, use a
  perceptually balanced diverging colormap centered exactly at zero.
- Include a colorbar with metric/units and report the color limits used.
- Do not independently autoscale comparable maps; show electrodes only when
  they aid interpretation and keep head outlines minimal.

### Subject-level session comparisons

- Show paired participant observations whenever possible.
- Connect repeated measurements from the same participant.
- Overlay a group summary rather than replacing individual observations with
  bars alone.
- Prefer distributions and an explicit mean/median with confidence intervals
  over bars for continuous measures.

### Statistical annotations

- Prefer exact p-values where practical.
- Do not use significance stars as the only statistical information.
- State what the central tendency, uncertainty interval/error bars, sample
  size, and statistical test represent in the figure or its legend.
- Clearly distinguish prespecified from exploratory analyses.
- Do not encode statistical significance using color alone.

## Figure legends and final quality control

Write figure legends in this order:

1. One-sentence description of the overall figure.
2. Description of panels in sequence (`a`, `b`, ...).
3. Definition of plotted quantities, lines, points, and error bands.
4. Sample sizes.
5. Statistical test and exact p-value or notation, when applicable.
6. Definitions of abbreviations that are not already obvious.

Before saving, verify:

1. Text remains readable at its final publication size.
2. Panel labels are visible and unclipped.
3. Labels do not overlap data or each other.
4. Axes include units where applicable.
5. Comparable panels use shared, scientifically justified limits.
6. Colors are accessible and consistent.
7. Statistical annotations are legible and informative.
8. The figure remains understandable after reduction to journal size.
