🧩 Stochastic Hierarchy Induction (SHI) for Time Series Classification

Author: Celal Alagoz
License: MIT
Python version: 3.10+
Last updated: November 2025

🌲 Overview

This repository provides the official implementation of the Stochastic Hierarchy Induction (SHI) framework — a classifier-informed approach for automatic hierarchy generation (HG) and hierarchy exploitation (HE) in Time Series Classification (TSC).

Unlike traditional hierarchical classification (HC) that relies on predefined label trees, SHI automatically derives hierarchical structures from flat label sets using the discriminative power of base classifiers.
The framework introduces Stochastic Splitting Functions (SSFs) — potr, srtr, and lsoo — to build diverse, data-driven hierarchies, and employs an extended Local Classifier Per Node (LCPN+) strategy for efficient exploitation.

<p align="center"> <img src="docs/figures/hierarchy_examples.png" alt="Examples of generated hierarchies" width="700"> <br> <em>Examples of hierarchies generated using SSFs: potr, srtr, and lsoo</em> </p>
🚀 Key Features

Automatic Hierarchy Generation (HG) via classifier-guided splitting

Three Stochastic Splitting Functions (SSFs):

potr — pick-one-then-regroup

srtr — split-randomly-then-regroup

lsoo — leave-salient-one-out

Hierarchy Exploitation (HE) using LCPN+ — an enhanced local hierarchical classifier

Plug-and-Play Classifiers: compatible with any scikit-learn or aeon estimator

Benchmark-ready: tested on 92 datasets from the UCR TSC Archive

Supports nested/flat evaluation and detailed runtime + performance breakdowns

🧠 Framework Overview

SHI operates in two main stages:

Hierarchy Generation (HG):

Classes are recursively split into sibling groups using one of the SSFs.

Each candidate split is evaluated by training a lightweight classifier on sibling groups.

The hierarchy that maximizes validation performance is selected.

Hierarchy Exploitation (HE):

Each parent node trains a local classifier (LCPN+) on its child groups.

Predictions traverse the tree top-down, combining decisions from all local models.
