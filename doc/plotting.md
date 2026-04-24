# Plotting Guide

This repository is optimized for publication-ready figures with a clean, restrained, Nature Neuroscience-style presentation.

## General style

- Prioritize clarity, readability, and direct interpretation over decorative styling.
- Keep figures visually clean and compact, but not cramped.
- Make the key scientific contrast immediately obvious:
  BCI vs control, pre vs post, distractor vs no distractor, or other analysis-specific contrasts.
- Avoid redundant visual elements that do not help answer the hypothesis.

## Typography and export settings

- Use a simple sans-serif font.
- Default Matplotlib publication settings should be:

```python
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
    "font.size": 8,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "lines.linewidth": 1.0,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})
```

- Save publication figures as PDF by default.
- Save figures into the repository `figures/` folder unless there is a strong reason to do otherwise.

## Layout

- Use `tight_layout()` for figure spacing.
- Figure width should be compact and deliberate, but wide enough that labels, data, and legends do not feel crowded.
- Multi-panel figures should have balanced spacing and aligned axes where appropriate.
- Do not let titles, axis labels, or legends collide with the plotting area.

## Legends

- Legends should not overlap data points, error bars, ERP traces, or topographic content.
- Prefer placing legends outside the plotting region when overlap is possible.
- Side placement is preferred when it preserves the data region and keeps the figure readable.
- Keep legend content minimal and consistent across figures.

## Axes

- Always include padding below the lowest plotted value and above the highest plotted value on the y-axis.
- No point, line endpoint, error bar, or marker should sit on the axis border or appear visually clipped.
- Set axis limits from the actual plotted content, not arbitrary defaults, unless a fixed scale is scientifically necessary for comparison.
- Keep axis labeling concise and informative.
- Use consistent axis ranges across panels when direct visual comparison is intended.

## Data display

- Show uncertainty clearly when relevant, typically with SEM or other explicitly defined error bars.
- Preserve subject-level information when it adds interpretability:
  thin paired lines and small points are appropriate when showing pre/post trajectories.
- Use restrained colors with clear group separation.
- Avoid heavy saturation, thick lines, or oversized markers unless needed for visibility.

## Nature Neuroscience-style preferences

- Minimal visual clutter.
- Clear contrast between conditions or groups.
- Professional, understated color palette.
- Thin axes and line work.
- Compact text and legends.
- Strong emphasis on scientific readability over stylistic novelty.

## Practical checks before finalizing a figure

- Are the main contrasts visually obvious?
- Does the legend avoid the data region?
- Is there visible y-axis padding above and below all plotted elements?
- Are labels readable at manuscript scale?
- Is the figure saved as a PDF in `figures/`?
- Would this figure still look clean if placed into a manuscript or slide without further cleanup?
