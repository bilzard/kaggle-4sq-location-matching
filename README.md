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

## Local Validation Result

- train/validation Split
  - split POIs 50:50
- train/validation set
  - drop rows with blank or nan `name`
  - drop POIs with a single location ID
  - sample selecting scheme
    - positive set
      - randomly sampled 2 samples from each POIs
    - negative set
      - randomly sampled 6 samples from nearest 25 samples excluding matched samples
        - distance metrics are calculated only on text (`name`, `categories`)
- Actual number of samples
  - 
### Blocker

<a href="https://ibb.co/TwX9Q40"><img src="https://i.ibb.co/sgS8TK1/validation-result-blocker.png" alt="validation-result-blocker" border="0"></a>
### Matcher

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