# Spike Daily Log

## Setup Information

**Hardware**:

- Computer: [Your machine]
- Camera: [Model/type]
- Resolution: [Actual resolution]

**Environment**:

- Lighting: [Description]
- Surface: [What cards sit on]
- Card sleeves: [Yes/No, what type]

**Photo of setup**: [Take and save as setup_photo.jpg]

---

## Day 1 - [DATE]

### Goals

- [ ] Python environment set up
- [ ] OpenCV installed
- [ ] Webcam feed displaying
- [ ] Hardware setup documented

### Completed

-

### Results

- Camera working: Yes/No
- Resolution achieved:
- FPS baseline:

### Issues

-

### Tomorrow

-

---

## Day 2 - [DATE]

### Goals

- [ ]
- [ ]

### Completed

-

### Results

-

### Issues

-

### Tomorrow

-

---

## Day 7 - WEEK 1 CHECKPOINT

### Summary Statistics

- Detection rate: \_\_\_%
- False positive rate: \_\_\_%
- Test images captured: \_\_\_/60

### On Track?

[ ] Yes [ ] No [ ] At Risk

### Adjustments Needed

-

---

## Day 14 - DECISION DAY

### Final Metrics

| Metric                  | Target | Actual | Pass/Fail |
| ----------------------- | ------ | ------ | --------- |
| Detection (good light)  | ≥95%   |        |           |
| Detection (typical)     | ≥85%   |        |           |
| Identification accuracy | ≥70%   |        |           |
| FPS                     | ≥15    |        |           |

### Decision: [ ] GO [ ] NO-GO [ ] CONDITIONAL

### Rationale

[Write 2-3 paragraphs explaining decision based on data]

```

## Step 10: Create .gitignore

Create file `.gitignore`:
```

# Python

venv/
**pycache**/
_.pyc
_.pyo
.Python
pip-log.txt

# IDE

.idea/
.vscode/
_.swp
_.swo

# Spike artifacts (large files)

spike/reference_cards/_.jpg
spike/reference_cards/_.png
spike/test_images/_.jpg
spike/test_images/_.png

# Keep the directories

!spike/reference_cards/.gitkeep
!spike/test_images/.gitkeep

# Results (track CSVs, not screenshots)

spike/results/failure_cases/

# OS

.DS_Store
Thumbs.db
