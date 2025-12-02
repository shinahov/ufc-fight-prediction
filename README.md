# UFC Fight Outcome Prediction

This project focuses on predicting the outcome of UFC fights using machine learning. The goal is to build a clean, realistic end-to-end pipeline that avoids common issues such as data leakage and incorrect historical reconstruction of fighter statistics.

## Motivation
Many UFC prediction projects use *current* fighter data (e.g., today's win/loss record, current reach, current age).  
This causes **data leakage**, because the model trains on information that did **not** exist at the time of the fight.

To avoid this, the pipeline reconstructs each fighter’s historical profile using only the information available **before** each event.

## Data Collection

### Event Scraper (`new_event_scraper`)
Collects all historical UFC events from UFC Stats:

- All fights in each event  
- Fighters involved  
- Basic statistics as they appeared on that event date  

This dataset is the primary training source, because it represents the fighters’ state *at the time of the fight*.

### Fighter Scraper
Collects current fighter profiles from UFC Stats.

This data is **not used** for model training because:

- It shows the fighter’s stats *today*, not historically  
- Using it would leak future information  

It is included only for analysis and completeness.

## Historical Reconstruction (`full_fighter_pipeline`)
This is the core of the project. It builds each fighter’s historical stats by accumulating data only up to the fight date.

Properties:

- No future data is used  
- Each fighter’s profile reflects exactly what was known *before* the fight  
- Prevents all major forms of data leakage  

This solves the primary issue found in many UFC prediction attempts.

## Modeling

Experiments are implemented in:

- `model_experiment`
- `xgb_train`

Explored:

- Multiple feature sets  
- Different preprocessing strategies  
- Baseline models and tree-based models  

## Best Result
Using **XGBoost**:

- **Test Accuracy:** ~66%  
- **AUC:** ~70%  

Despite the randomness of MMA outcomes, the model captures meaningful predictive patterns.

## Summary
This project builds a realistic UFC prediction pipeline by reconstructing historical fighter data and eliminating leakage, resulting in a solid and reliable predictive model.
