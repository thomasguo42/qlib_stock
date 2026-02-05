# LightGBM
* Code: [https://github.com/microsoft/LightGBM](https://github.com/microsoft/LightGBM)
* Paper: LightGBM: A Highly Efficient Gradient Boosting
Decision Tree. [https://proceedings.neurips.cc/paper/2017/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf](https://proceedings.neurips.cc/paper/2017/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf).


# Introductions about the settings/configs.

`workflow_config_lightgbm_multi_freq.yaml`
- It uses data sources of different frequencies (i.e. multiple frequencies) for daily prediction.

`workflow_config_lightgbm_Alpha158_us_sharadar_weekly_pit_best.yaml`
- Weekly US Sharadar PIT pipeline (Alpha158 + PIT fundamentals + ratio features) with a WeeklyTopkDropout strategy.
- This is the canonical config for the Sharadar US weekly pipeline in this repo.
