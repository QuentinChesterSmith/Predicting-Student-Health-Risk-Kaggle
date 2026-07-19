# Predicting-Student-Health-Risk-Kaggle
My attempts and work for season 6 episode 7 of the Kaggle Playground series (https://www.kaggle.com/competitions/playground-series-s6e7)

## Log
|Model/Change|CV (Balanced Accuracy)|Public Leaderboard|
|---|---|---|
|Baseline LGBM Model|0.94965|0.94995|
|Sleep Duration Features|0.94965|0.94995|


## Feature Engineering 
The first thing I did for feature engineering was I wrote a script to plot my target variable (health condition) against all of my features, specifically numerical features to look for patterns.
Using the feature_distribution notebook I genered 3 graphs per feature, one for each category in target (at-risk, unhealthy, and fit) and after looking at them I tried to find patterns and ideas for new features.
The main pattern I found was in sleep duration as there seemed to a huge difference in distrubtions between the three categories as shown in the images below.
![example](./Images/sleep_duration.png)

While I did try adding new three new binary featues, sleep<6, 6<sleep<7, and sleep>7 adding them did nothing to either the CV or the public leaderboard score. The code I used to create these new features is shown below.

```py
for df in [training_data, test_data]:
    df["sleep<=6"] = (df["sleep_duration"]<=6).astype(int)
    df["7=<sleep<=6"] = np.where((df["sleep_duration"] >= 6) & (df["sleep_duration"] <= 7), 1, 0)
    df["sleep>=7"] = (df["sleep_duration"]>=7).astype(int)
    df.drop(columns=["sleep_duration"])
```
