# The Whole Project, Explained Simply

This guide explains every step of `LinearRegression_Alerts.ipynb` in plain language. No maths
background is needed. Real numbers from the project are used wherever possible.

---

## 1. The story in one paragraph

A factory robot has eight joints. Each joint has a motor, and every 2 seconds the robot reports
how much electricity (current) each motor is drawing. When a joint starts to wear out, its motor
has to work harder, so it draws **more current than usual, for longer than usual**. This project
teaches the computer what "usual" looks like for each joint, then sets up two alarms:

- a yellow **Alert** ("keep an eye on this"), and
- a red **Error** ("stop and check now").

Both alarms go off only if the extra current **lasts** for a while, not just for a single blip.

---

## 2. The data

| Thing | What it means |
|---|---|
| **39,672 readings** | rows in the CSV file, one every ~2 seconds for 22.5 hours |
| **Axis #1 … Axis #8** | the eight joints of the robot |
| **Current (amps, A)** | how hard each joint's motor is working right now |
| **Time** | when the reading was taken |

**Think of it like this:** a fitness tracker records your heart rate every few seconds. The robot
records "motor effort" for eight joints every 2 seconds.

> **Why not kWh?** The assignment talks about kWh (energy). Energy needs voltage *and* current, and
> this file only has current. So every threshold is in **amps**.

### Where the data lives

The data is stored in a **cloud database** (Neon PostgreSQL), a spreadsheet that lives on the
internet. The notebook connects to it and downloads all the readings with one function,
`fetch_all()`. It's the same data as the CSV, but reading it from the database is how a real
system would work.

---

## 3. Splitting the data: practice vs check

The readings are split **in time order**:

- **Training (first 80% ≈ 18 hours):** the computer learns from this part.
- **Calibration (last 20% ≈ 4.4 hours):** used to check the rules on data the model hasn't seen.

**Why not shuffle?** In real life you only ever learn from the past and then face the future.
Shuffling would let the model "peek" at later data.

---

## 4. The idle problem (an important decision)

About **two-thirds** of readings are all zeros: the robot is parked, so every motor reads 0 A.

Imagine working out a footballer's "normal" running speed while including all the time they spent
sitting on the bench. The average comes out very low, and every time they actually run it looks
"too fast".

The same thing happened here: the zeros dragged the "normal" line down, so the robot simply
*moving* looked abnormal. **The fix:** learn "normal" from **active** readings only (at least one
joint drawing current). Parked readings can never set off an alarm, because zero current is never
*above* normal.

---

## 5. What is linear regression?

Linear regression draws **the best straight line through a cloud of dots**.

```text
current ▲
        │   •     •        •
        │ •   •  •   •  •     •
        │─────────────────────────  ← the line = "normal current"
        │  •    •   •    •  •
        │•   •      •  •     •
        └────────────────────────▶ time
```

A straight line has two numbers:

| Name | Plain meaning | Example (Axis 2) |
|---|---|---|
| **Intercept** | where the line starts, i.e. normal current at time zero | 10.03 A |
| **Slope** | how much the line rises or falls each hour | +0.002 A per hour |

So for Axis 2 the model says: **normal current ≈ 10.03 A, and it is barely changing over time.**

"**Univariate**" just means there is **one input** (time). There are eight separate lines, one per
joint.

### How does the computer pick "the best" line?

It tries to make the line as close to all the dots as possible. For each dot it measures the
vertical gap to the line, **squares** it (so gaps above and below don't cancel out, and big misses
count extra), and adds them all up. The best line is the one with the **smallest total**. This is
called **least squares**.

### Standardizing first (z-scores)

Time is measured in tens of thousands of seconds, while current is measured in single amps. To keep
the maths tidy, both are converted to **z-scores** before fitting:

```text
z = (value − average) ÷ standard deviation
```

A z-score says **how many "typical wobbles" a value sits from the average**.

**Example:** a joint averages 10 A with a typical wobble (standard deviation) of 2 A. A reading of
14 A has z = (14 − 10) ÷ 2 = **2**, so it is two wobbles above average.

The averages and wobbles are learned from **training data only** and reused for everything after
that. This is like grading every exam with the same ruler instead of making a new ruler for each
exam.

---

## 6. Predictions and residuals: "how wrong was the line?"

For every reading, the model predicts what the current *should* be. Then:

```text
residual = actual current − predicted current
```

| Actual | Predicted | Residual | Meaning |
|---|---|---|---|
| 14 A | 10 A | **+4 A** | drawing 4 A **more** than normal ⚠️ |
| 10 A | 10 A | 0 A | exactly normal |
| 7 A | 10 A | −3 A | drawing less than normal (not a worry) |

**Only positive residuals matter for the alarms**, because a worn joint works *harder*, not less.

---

## 7. How good are the lines?

| Score | Plain meaning |
|---|---|
| **MAE** | average size of the miss, in amps |
| **RMSE** | like MAE, but punishes big misses more |
| **R²** | how much of the up-and-down the line explains (1 = all of it, 0 = none) |

**R² is about 0 for every joint.** That sounds bad, but it isn't here. Within a single day, a healthy
robot's current doesn't trend up or down, so *time* can't explain the moment-to-moment ups and
downs. Those come from the robot's work cycle (lift, move, drop, repeat).

What the near-zero slope **does** tell us is useful: **no joint got worse during these 22 hours.**
The line works as a "normal level" baseline, which is exactly what the alarms need.

---

## 8. Looking at the residuals

The notebook draws four kinds of pictures:

1. **Histograms:** how common each size of miss is. They show a long tail on the right: lots of
   small misses, plus occasional big jumps above the line. Those jumps are the robot doing heavy
   lifts.
2. **Residuals over time:** a flat band means nothing is slowly creeping up. A worn joint would show
   the band drifting upward.
3. **Box plots:** each box holds the middle half of the misses, and the dots beyond it are
   "outliers". Training and calibration look the same, so the later hours behave like the earlier
   ones.
4. **Outlier counts:** two textbook rules (IQR and "3 standard deviations") flag roughly 2–13% of
   all working readings. That is far too many to all be faults: they're normal heavy moves.

**Lesson:** you cannot spot a fault by size alone. A healthy robot regularly has big, short spikes.
What makes a fault different is that the extra current **doesn't go away**.

---

## 9. Finding the thresholds (the most important part)

The rules need three numbers:

| Name | Question it answers |
|---|---|
| **MinC** | How far above normal counts as worrying? (Alert level) |
| **MaxC** | How far above normal counts as serious? (Error level) |
| **T** | How long must it last? (seconds) |

### Step 1: candidate levels from percentiles

A **percentile** ranks the misses. The **95th percentile (p95)** is the level that only 5% of
working readings ever reach. The notebook lists p80, p85, p90, p95, p97.5, p99 and p99.5 for each
joint.

### Step 2: how long do healthy spikes last?

Every time a joint went above its p95 level, the notebook counted how many readings in a row it
stayed there:

| Spike length | How many times (all joints, training) |
|---|---|
| 1 reading | 3,732 |
| 2 readings | 238 |
| 3 readings | 8 |
| **4 or more readings** | **0** |

So **healthy spikes are blips**: 93.8% vanish after a single reading, and none lasts four.

### Step 3: try every combination

Every level (p80 … p99.5) was combined with every duration (2, 4, 5, 6 and 10 s) and checked
against the healthy calibration data. The table counts how many alarms would have gone off:

| Level / time | 2 s | 4 s | **5 s** | 6 s | 10 s |
|---|---|---|---|---|---|
| p80 | 156 | 27 | 27 | 11 | 0 |
| p85 | 82 | 5 | 5 | 2 | 0 |
| p90 | 33 | 1 | 1 | 0 | 0 |
| **p95** | 3 | 0 | **0** | 0 | 0 |
| p99 | 0 | 0 | 0 | 0 | 0 |

The robot is healthy, so a good rule should give **0** here. Lower levels or shorter times create
false alarms, and false alarms teach people to ignore alarms. **p95 with 5 seconds is the first
point where healthy data stays completely quiet.**

### Step 4: the final rules

- **MinC = p95**, the lowest level that stays quiet on healthy data.
- **MaxC = p99**: only 1 in 100 working readings gets that high, even for an instant.
- **T = 5 seconds.** Readings arrive every ~2 s, so 5 s means **four high readings in a row**,
  which never happens in healthy data.

**Example, Axis 2:** normal ≈ 10.0 A and MinC = 16.0 A, so an **Alert** needs current at or above
**≈ 26 A for 5 seconds**. MaxC = 31.9 A, so an **Error** needs **≈ 42 A for 5 seconds**.

### The Axis 7 special case

Axis 7 never goes above **8.11 A**; it seems to have a built-in limit. Because of that ceiling, its
p95 and p99 are almost identical (just 0.07 A apart), so Alert and Error would mean the same thing.
The fix is a simple extra rule: **MaxC must be at least MinC + one standard deviation.** For Axis 7,
an Error now means "drawing more than this joint has ever drawn", which is a sensible meaning for
an Error. No other axis is affected.

### Gaps in the data

If two readings are more than 5.7 seconds apart (a hiccup in recording), the timers restart. The
system doesn't pretend to know what happened during the gap.

---

## 10. Making test data (synthetic data)

To test the alarms, new data was needed. The assignment suggests asking an AI to invent a file with
similar statistics. Instead, the notebook builds it in code, because that's **repeatable** (the same
seed gives the same data every time) and **more realistic**.

### Healthy background (4,000 readings)

Random **chunks of 30 consecutive real readings** are copied from the training data and stitched
together. Like making a new playlist from 30-second clips of real songs, each chunk still sounds
real, and the joints still move together naturally.

### Checking it matches (normalization and standardization)

Two checks confirm the test data looks like the training data **measured with the training data's
own ruler**:

- **Min-max normalization:** squash values onto a 0-to-1 scale using the training minimum and
  maximum. No healthy test reading went above 1, so nothing is outside what training ever saw.
- **Z-scores:** count what share of readings sit within 1, 2 and 3 "wobbles" of the training
  average. Training and test shares match within about 1 percentage point on most joints.

### Fake faults with known answers

Faults were then planted into every joint, like a fire drill where you already know which alarms
*should* ring:

| Planted fault | Should the alarm ring? |
|---|---|
| a 4-second spike | ❌ no, too short |
| 8 seconds between MinC and MaxC | 🟡 Alert |
| 8 seconds above MaxC | 🟡 Alert + 🔴 Error |
| high, but interrupted by one normal reading | ❌ no, the timer resets |
| high, but split by a gap in recording | ❌ no, the timer resets |
| load slowly climbing for a minute (like wear) | 🟡 Alert first, then 🔴 Error |

**Result: 48 of 48 drills behaved exactly as expected.**

---

## 11. Streaming through the database

A real robot doesn't hand over a finished file. Readings arrive one at a time. To simulate that:

1. The **streaming simulator** releases the test file **one reading at a time** (100× faster than
   real life, to save waiting).
2. Each reading is **saved into Neon** (table `synthetic_test_readings`).
3. The notebook **reads it back from Neon**, the way a monitoring app would.
4. The model **predicts** the normal current and calculates the residual.
5. The **live detector** updates its timers.

### The live detector: two stopwatches per joint

For every joint, picture two stopwatches:

- 🟡 The **Alert stopwatch** runs while current is ≥ MinC above normal.
- 🔴 The **Error stopwatch** runs while current is ≥ MaxC above normal.

When current drops back below the level, that stopwatch **resets to zero**. When a stopwatch
reaches **5 seconds**, the alarm goes off **once**. The detector remembers its stopwatches between
batches of readings, so a fault that starts at the end of one batch and continues into the next is
still timed correctly.

### Double-checking

After streaming, the notebook proves three things:

- the data that came back from Neon matches the original file exactly;
- the live alarms match a full recalculation done afterwards;
- the results match a calculation done without the database at all.

---

## 12. Logging the alarms

Every alarm is saved with:

- which **joint** and which **level** (Alert or Error);
- **when** it started and when it was confirmed;
- **how long** it lasted;
- **how far** above normal it peaked.

It's saved in two places: `results/events.csv` (a spreadsheet) and the Neon table `alert_events`
(the shared log a maintenance dashboard would read).

**Last run: 40 alarms (24 Alerts, 16 Errors). Every one came from a planted fault, and the healthy
data caused none.**

---

## 13. Reading the charts

| Chart | What to look for |
|---|---|
| **Regression lines** (8 panels) | blue = training, orange = calibration, dark line = "normal" |
| **Residual histograms** | the long right tail = normal heavy moves |
| **Residuals over time** | a flat band = no slow wear |
| **Box plots** | training and calibration boxes look alike = stable robot |
| **Run lengths** | huge bar at "1 reading", nothing at "4+" = healthy spikes are blips |
| **Threshold discovery** | where the lines hit zero = levels that stay quiet on healthy data |
| **Per-axis event charts** | 🟡 triangles = Alerts, 🔴 diamonds = Errors, each labelled with its duration (e.g. "Error · 30.0 s") |

---

## 14. What it all means for maintenance

- The robot looks **healthy**: no drift in 22 hours, and the alarms stay quiet on real data.
- The alarms catch **sustained** extra effort within about 6 seconds, which is the signature of a
  joint that is binding or wearing.
- 🟡 **Alert** = book an inspection at the next planned stop.
  🔴 **Error** = stop the robot and check now.
- Long-term, saving the lines every week and watching whether the **slope starts rising** (especially
  on the hardest-working joints, Axes 2 and 3) would give months of warning before a breakdown.

---

## 15. Honest limitations

- **No real faults in the data**, so we can't measure how often the alarms would be right on a
  real failure.
- **The calibration data helped choose the rules**, so it isn't a completely independent test.
- **The test data is made from training data.** It proves the alarm logic works, not real-world
  accuracy.
- **One day is too short to see wear**, which happens over weeks and months.
- **One edge case:** two high readings with a long recording gap between them can just meet the
  5-second rule. Also requiring a minimum number of readings would fix it.

---

## Mini glossary

| Word | Meaning |
|---|---|
| **Current (A)** | how hard a motor is working |
| **Regression line** | the best straight line through the dots = "normal" |
| **Intercept** | where the line starts |
| **Slope** | how fast the line rises or falls |
| **Residual** | actual − predicted: how far off normal a reading is |
| **Percentile (p95)** | the level only 5% of readings reach |
| **Standard deviation (SD)** | the typical size of a wobble around the average |
| **Z-score** | how many SDs a value is from the average |
| **Min-max normalization** | rescaling values to a 0-to-1 range |
| **MinC / MaxC** | the Alert / Error levels, in amps above normal |
| **T** | how many seconds the extra current must last |
| **Calibration set** | the later data used to test and tune the rules |
| **Synthetic data** | test data made by the computer |
| **Neon / PostgreSQL** | the cloud database storing the readings and the alarms |
