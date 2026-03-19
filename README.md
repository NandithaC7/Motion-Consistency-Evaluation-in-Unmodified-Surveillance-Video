# Motion Consistency Evaluation in Unmodified Surveillance Video

## Project Type

Final Year Project (Research-Based)
Course: 23CSE399 – Project Phase 1

---

## Problem Statement

Surveillance camera systems generate continuous video streams; however, real-world feeds often suffer from hidden issues such as frozen frames, prolonged low motion, and temporal inconsistencies caused by environmental or hardware conditions. These issues do not trigger traditional object-based alerts but significantly reduce monitoring reliability.

This project focuses on evaluating motion consistency in unmodified surveillance videos to assess camera feed health.

---

## Objective

The objective of this work is to develop a camera-centric evaluation framework that:

* Analyzes temporal motion patterns in surveillance videos
* Detects silent failures such as freezing, stagnation, and instability
* Produces a quantitative Feed Reliability Score

---

## Research Gap

Existing approaches primarily focus on:

* Scene-level anomaly detection (behavioral events)
* Video forensics (intentional tampering detection)
* Video quality assessment (perceptual distortion measurement)

These methods assume reliable camera feeds and do not evaluate feed-level reliability directly. There is no unified framework that assesses temporal motion consistency for real-time camera health monitoring.

---

## Proposed Approach

### Motion Modeling

* Frame differencing
* Optical flow estimation

### Temporal Analysis

* Sliding window statistical modeling
* Mean and variance-based consistency evaluation

### Detection Strategy

* Frozen frame detection
* Motion stagnation detection
* Temporal discontinuity identification

### Output

* Feed Reliability Score representing camera health

---

## Mathematical Formulation

Motion Estimation:
Mₜ = ||Fₜ − Fₜ₋₁||

Consistency Measure:
Cₜ = |Mₜ − μ| / (σ + ε)

Reliability Score:
R = 1 − (1/T) Σ I(Cₜ > τ)

---

## Methodology Pipeline

1. Spatio-temporal feature extraction from video frames
2. Motion representation using frame differencing and optical flow
3. Sliding window-based temporal consistency evaluation
4. Detection of deviations (freeze, duplication, stagnation)
5. Computation of Feed Reliability Score

---

## Dataset

ShanghaiTech Campus Surveillance Dataset

* Real-world surveillance footage
* Static camera setup
* Natural motion patterns across multiple scenes

---

## Repository Structure

* `experiments/` – motion extraction and analysis scripts
* `results/` – generated outputs and observations
* `literature/` – paper summaries and comparisons
* `documentation/` – project documentation

---

## Current Status (Phase 1)

* Literature survey completed
* Research gap identified
* Conceptual model and architecture defined

## In Progress (Phase 1)

* Initial motion analysis experiments in progress
* Improving our methodology
* Finalizing model that is to be implemented

## Team
* Nanditha C
* Harisree M
* Yartha Vinutha
* Krishna Veni Valluri

## Guide
Dr. N Lalithamani
Department of Computer Science and Engineering
Amrita School of Computing
