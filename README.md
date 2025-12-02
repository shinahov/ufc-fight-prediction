UFC Fight Outcome Prediction

This project focuses on predicting the outcome of UFC fights using machine learning.
The goal was to build a clean, realistic end-to-end pipeline that avoids common mistakes such as data leakage and incorrect historical reconstruction of fighter statistics.

Motivation

Most UFC prediction projects online use the current fighter data (e.g., current win/loss, current age, current reach).
This creates data leakage, because you train on information that did not exist at the time of the fight.

To solve this, I built a pipeline that reconstructs historical fighter data, based only on information available at the date of each event.

Data Collection
1. Event Scraper (new_event_scraper)

This scraper collects all historical UFC events from the UFC Stats website.
For every event, it extracts:

Fights

Fighters involved

Basic statistics from that specific event date

This is the main data source used for training, because it reflects the fighters' state at the time of each fight.

2. Fighter Scraper

This scraper collects current fighter profiles.

However, this data cannot be used for training a historical model because:

It shows the fighter's stats today, not at the time of past fights

Using it would leak future information into the training set

Therefore, this scraper is included for completeness, but not used in the final model pipeline.

Historical Reconstruction (full_fighter_pipeline)

The full_fighter_pipeline is the core part of the project.
It accumulates fighter statistics only up to the date of each fight, ensuring that:

No future data is included

Each fighter’s profile reflects their experience up to that event

The model sees only “realistic” historical features

This step solves the major problem of data leakage that appeared in earlier attempts.

Modeling

All experiments are implemented in:

model_experiment

xgb_train

Several approaches were tested:

Different feature sets

Different preprocessing strategies

Multiple ML models (baseline models, tree models, etc.)

Best Result

The best performance was achieved using XGBoost:

Test Accuracy: ~66%

AUC: ~70%

This indicates that the model is able to capture meaningful patterns, despite the natural randomness and unpredictability of UFC fights.
