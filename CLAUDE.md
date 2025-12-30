# Card Detection Spike - Project Conventions

## Project Type

This is a RESEARCH SPIKE, not production code. The goal is to answer questions, not build features.

## Time Box

2 weeks maximum. Decision point: Day 14.

## Primary Question

Can we reliably detect and identify MTG cards from a webcam feed?

## Pass/Fail Criteria

### Detection (Finding Cards)

| Metric                         | Pass | Fail |
| ------------------------------ | ---- | ---- |
| Detection rate (good lighting) | ≥95% | <85% |
| Detection rate (typical room)  | ≥85% | <70% |
| Sleeved cards                  | ≥90% | <75% |

### Identification (Which Card)

| Metric                   | Pass   | Fail    |
| ------------------------ | ------ | ------- |
| Accuracy (good lighting) | ≥85%   | <70%    |
| Accuracy (typical room)  | ≥70%   | <55%    |
| Time to identify         | ≤500ms | >1000ms |

### Performance

| Metric         | Pass   | Fail |
| -------------- | ------ | ---- |
| Processing FPS | ≥15    | <10  |
| Memory usage   | ≤500MB | >1GB |

## Tech Stack

- Python 3.11+
- OpenCV 4.9+
- NumPy
- Requests (for Scryfall API)

## Code Style

- Single-file scripts are acceptable (this is a spike)
- Document findings, not code elegance
- Console output for debugging
- CSV logging for quantitative results

## What NOT To Build

- No UI frameworks
- No web servers
- No databases
- No Angular/React/Electron
- No "clean architecture"

## Daily Workflow

1. Update DAILY_LOG.md with goals
2. Do the work
3. Document results quantitatively
4. Plan tomorrow

## Definition of Done (for spike)

- [ ] Detection rate measured across 60 test images
- [ ] Identification accuracy measured
- [ ] Performance benchmarked
- [ ] Failure cases documented with screenshots
- [ ] Go/No-Go recommendation written with data

## Commands

```bash
# Setup
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Run
python spike/spike.py
```
