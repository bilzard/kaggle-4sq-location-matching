# Kaggle foursquare location matching

This repository is source code of my solution in a Kaggle competition [Foursquare - Location Matching].

[Foursquare - Location Matching]: https://www.kaggle.com/competitions/foursquare-location-matching

## Summary of Solution

1. Blocker - text(SBERT), location, combination of text & location
    - text
      - weighted concatenation of SBERT embeddings of `name` and `categories` (384 dimmensions for each)
      - optimal weights are searched with [Optuna]
    - location
      - haversine distance (2 dimensions)
    - combination
      - weighted concatenation of
        - text: SBERT embeddings on `name` and `categories` (384 dimmensions for each)
        - location: cartesian coorinate (3 dimentions)
        - optimal weights are searched with [Optuna]
    - for scalability, I divided search space using [H3] geospatial index
2. Matcher - binary classifier backed by BERT
    - fine-tuning BERT model (all-MiniLM-L6-v2)

[H3]: https://github.com/uber/h3
[Optuna]: https://github.com/optuna/optuna

## Train/Validation Set for the Matcher

- train/validation Split
  - split POIs 50:50
- preprocess
  - drop rows with blank or nan `name`
  - drop POIs with a single location ID
- pair selecting scheme
  - positive pair
    - randomly sampled 1 samples from each POIs, making 1 pair per POI
  - negative pair
    - randomly sampled 6 samples from nearest 25 samples excluding matched samples, making 6 pairs per POI
      - distance metrics are calculated only on text (`name`, `categories`)
- actual number of samples
  - train: 1,346,336
  - test: 1,344,027

## Local Validation Result
### Blocker

The maximum Recall=0.9825 is achieved by n_neighbor=85.695.

<a href="https://ibb.co/Ln7vsSm"><img src="https://i.ibb.co/GcSFr7y/validation-result-blocker.png" alt="validation-result-blocker" border="0"></a>
### Matcher

**Note**: The below W&B graph is evaluation result of a tiny subset of test set (~5K). For the full-set evaluation result is IoU=**0.89393**

```
val/f1=0.94399, val/recall=0.94882, val/precision=0.93921, val/iou=0.89393, val/threshold=0.4000
```


<a href="https://ibb.co/ykb1GTh"><img src="https://i.ibb.co/wQxX3HJ/validation-result-matcher.png" alt="validation-result-matcher" border="0"></a>

## Inference Notebooks

- [combination, k=25](https://www.kaggle.com/code/tatamikenn/4sq-submit-combination-offline)
- [union(text, location, combination), k=(18, 5, 7)](https://www.kaggle.com/code/tatamikenn/4sq-submit-union-c-t-l-18-5-7-offline)

Note: they are **not ranked** because of Notebook Error (probably memory overflow).

## Training

```console
$ train_ditto.py /content/data/ditto_dev_feat_all_n25_seed1234_full.tsv.gz /content/data/ditto_test_feat_all_n25_seed1234_small.tsv.gz --finetuning --batch_size 64 --lr 3e-5 --fp16 --save_model --lm all-MiniLM-L6-v2 --n_epochs 10 --run_id 0
```

## ToDo

- [x] include inference code
- [x] include training code
- [ ] include code to generate training dataset for matcher
## Attribution

This repository is based on the code of [Ditto].
However, I didn't apply any tricks used in Ditto (e.g. knowledge injection, sentence summation, augmentation etc.). I used ditto just fine-tuning the BERT model. Since Ditto assumes the input text is English, it can't be adopted to 4sq competition straightforward, where the input text is multi-lingual. In particular, in some Asian languages (Japanese, Chinese etc.), tokens are not space-separated, but ditto assumes that they are space-separated.

[Ditto]: https://github.com/megagonlabs/ditto