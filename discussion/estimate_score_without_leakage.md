# How to estimate the score without (trivial) leakage?

I share the way to estimate the LB score without trivial leakage.
The definition of trivial leakage is: "the triplet (`name`, `latitude`, `longitude`) are exactly same" (see the discussion [2]).
I didn't considered any non-trivial leakage in the discussion below.

Note: in the discussion below, the LB score means **private LB score** (However, we can discuss the similar discussion for the public LB score).

# Step1: Estimate the score of the locations with leakage

We already have the fact below (c.f. the discussion [2]):

* Using my submission utilizing leakage[1], the score is 0.884
* The ratio of leaked row estimated in the above prediction is 0.67

## Estimation

Suppose,

* `s_1`: the LB score utilizing leakage (=0.884) [1]
* `s_0`: the LB score of trivial submission (=0.650) [3]
* `s_leak`: the LB score utilizing leakage for 67% leaked rows (the value to estimate)

Using the above notation, we have the equation:

$$s_1 = s_\text{leak} * 0.67 + s_0 * 0.33 \tag{1}$$

Solving the below equation (1), we obtain

$$s_\text{leak} = \frac{s_1 - s_0 * 0.33}{0.67}=0.9997 \tag{2}$$

(It is almost perfect score!)

# Step 2: Estimate the score of your model without trivial leakage

1. Submit the notebook with the below procedure and obserb the score (say s_2).
   1. in 67% of leaked rows, set the predictions using notebook [1].
   2. in 33% of non-leaked rows, set the predictions using your model.

Using the value of s_leak and s2, we have an below equation:

$$s_2 = s_\text{leak} * 0.67 + s_\text{non-leak} * 0.33 \tag{3} $$

By the above equation, we obtain,

$$s_\text{non-leak} = \frac{s_2 - s_\text{leak} * 0.67}{0.33} \tag{4}$$

E.g. if your s_2 = 0.977 (same as 1st teams score), s_non_leak = **0.9309**.

# Reference

* [1] https://www.kaggle.com/code/tatamikenn/4sq-leakage-submission-v3
* [2] https://www.kaggle.com/competitions/foursquare-location-matching/discussion/336047
* [3] https://www.kaggle.com/code/tatamikenn/4sq-trivial-submission/notebook