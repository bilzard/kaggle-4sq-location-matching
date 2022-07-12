# How should we have noticed leak during competition?

## Summary

I synthesized simulated leaked test data from train.csv, and tested how we could have noticed the potential leaks during the competition.

The basic concept is comparing the score of cross fold models and that of trained by all-data (for convenience, we will call them *N-fold(s)* and *all-data* respectively).
Since exactly one of *N-fold* models doesn't see part of train data, averaging predictions can utilize only (N-1)/N of leaked information. On the other hand, *all-data* model can utilize 100% of leaked information. With that reason, the LB score of *all-data* should be higher than that of *N-folds* suppose test data contains leaked information.

## Experimental Settings

I also simulated this phenomena using the competition data.
The experiment condition is as below:
1. make 1/16 subset of train.csv (lets call this *sim*)
1. split *sim* into 50:50 to make *develop* and *test*
1. split *develop* into 50:50 to make fold0 and fold1

Note that in all of the above process, I splitted data based on POIs to avoid leaks.

Next, I made *test_leaked* in below process:
1. split *sim* into 50:50 to make *test_leaked* (with different random seed than I used splitting develop and test)

Using these datasets, I compared the IoU scores of a) *all-data* model and b) *n-fold* models over 1) test(non-leak), 2) test(leak=0.5), and 3) test(leak=1.0).

I shared the notebook which is used to generated these experiment data in [1, 2].

## Experimental Result

The calculated IoU scores is shown in Fig1. Note that all scores are averaged by 3 models with different random seeds.
If the leakage ratio increases, the difference of IoU between *all-data* and *n-fold* increases.
In particular, if the leak ratio is 0.5, the difference of IoU is about 2.0%.

<center><b>Fig1: Result of Leakage Simulation</b></center>
<a href="https://ibb.co/DpqDh4W"><img src="https://i.ibb.co/0twjTq2/Screen-Shot-2022-07-11-at-15-09-03.png" alt="Screen-Shot-2022-07-11-at-15-09-03" border="0"></a>

## Discussion

Since the competition's test data estimated to contain 0.67 of leakage information[3], 
if we have compared the score of *all-data* and *n-fold*,  the observed score difference should have been at least **2.0%**.
If you have enough confidence that this score discrepancy is anomaly, you should have noticed the leak.

## Acknowledgements

The original idea is discussed in twitter by @columbia2131 and @onodera  . I would thank these two Kagglers.


## Reference

- [1] https://www.kaggle.com/tatamikenn/4sq-knn-for-training-data-leak-simulation
- [2] https://www.kaggle.com/code/tatamikenn/4sq-make-ditto-input-leak/notebook
- [3] https://www.kaggle.com/competitions/foursquare-location-matching/discussion/336047