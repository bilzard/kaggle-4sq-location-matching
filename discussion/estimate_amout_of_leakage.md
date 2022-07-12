# LB Probe Result on Leakage - About 67.0% of Rows are Leaked

## Summary

* leakage is defined as: "the triplet (`name`, `longitude`, `latitude`) are exactly same".
* About  ~~33.0 %~~ **67.0%** of rows in the `test.csv` also exis in `train.csv`
* A submission only using leaked information could score at least 0.884 [1]

## Detail

I estimated the ratio of leakage exists in test.
The estimation is done by simply comparing to LB scores of reference point of x in [0, 1], where submission is done by filling `matches` with `id` if `index < x * len(test)`.
The regression coefficient of the regression curve by the least squares method is 1.00 with a slope of 1.537448 and an intercept of 0.001131.
(The notebook used for probing is shared in [2, 3].)

<a href="https://ibb.co/0XYYdqW"><img src="https://i.ibb.co/dckkzB3/Screen-Shot-2022-07-10-at-14-35-42.png" alt="Screen-Shot-2022-07-10-at-14-35-42" border="0"></a>

## Reference

* [1] https://www.kaggle.com/code/tatamikenn/4sq-leakage-submission-v3
* [2] https://www.kaggle.com/code/tatamikenn/4sq-leakage-estimation
* [3] https://www.kaggle.com/code/tatamikenn/4sq-reference-submission-for-probing/notebook