# Switching change-point detection modes

The Streamlit app and the headless runner now share the same detection pipeline, so you can swap between the **classifier** (UI-style) and **simple** detectors just by changing configuration.

## Detector modes at a glance

| Mode | `DetectionConfig.use_classifier` | What it does |
| --- | --- | --- |
| **Classifier** | `True` | Runs the logistic-change detector, classifies each breakpoint (transient vs. persistent, rate vs. level/mixed), and—when requested—filters to persistent rate/mixed shifts before seeding events. Matches the Streamlit experience. |
| **Simple** | `False` | Uses the lightweight `detect_change_points` helper with no classification step. Good for exploratory runs where you want every structural break without semantic filtering. |

Other knobs live on the shared `DetectionConfig`: `max_changes`, `min_seg_len`, `penalty_scale`, `window`, `z_pulse`, `rate_factor`, and `level_factor`. Larger penalties and windows smooth the results; smaller minimum segment lengths make the detector more sensitive.

## Streamlit app

When you click **Detect change dates** the app builds a `DetectionConfig(use_classifier=True, max_changes=<slider value>, window=6)` and filters to persistent rate/mixed breakpoints before writing to session state. To experiment with the simple detector inside the UI, adjust the config near `events_editor` in `app.py` (search for `DetectionConfig(`) and set `use_classifier=False`.

## Headless runner (`scripts/run_headless.py`)

The CLI exposes the same controls so you can flip modes without editing code:

```bash
# Classifier mode (default): matches the Streamlit behaviour
python scripts/run_headless.py \
  --all total.csv --all-has-header --all-date-col date --all-count-col total \
  --paid paid.csv --paid-has-header --paid-date-col date --paid-count-col paid \
  --detector classifier --max-changes 4 --window 6 --out-dir outputs/classifier

# Simple detector: skips classification and persistence filtering
python scripts/run_headless.py \
  --all total.csv --all-has-header --all-date-col date --all-count-col total \
  --paid paid.csv --paid-has-header --paid-date-col date --paid-count-col paid \
  --detector simple --min-seg-len 3 --penalty-scale 2.5 --out-dir outputs/simple
```

Additional flags:

- `--keep-all-breaks`: keep every breakpoint even in classifier mode (skip the persistent rate/mixed filter).
- `--min-seg-len`, `--penalty-scale`: mirror the kwargs that previously lived in the standalone detector.
- `--window`, `--z-pulse`, `--rate-factor`, `--level-factor`: tune classification sensitivity when `--detector classifier` is active.

Because both entry points call `run_detection`, switching modes in one place produces the same breakpoint dates everywhere, assuming the rest of the settings match.
