# Kaggle foursquare location matching

This repository is source code of my solution in a Kaggle competition [Foursquare - Location Matching].

[Foursquare - Location Matching]: https://www.kaggle.com/competitions/foursquare-location-matching

## Summary of Solution

1. Blocker - text(SBERT), location, combination of text & location
    - for scalability, I divided search space using [H3] geospatial index
2. Matcher - binary classifier backed by BERT (all-MiniLM-L6-v2)

[H3]: https://github.com/uber/h3

## Inference Notebooks

- [combination, k=25](https://www.kaggle.com/code/tatamikenn/4sq-submit-combination-offline)
- [union(text, location, combination), k=(18, 5, 7)](https://www.kaggle.com/code/tatamikenn/4sq-submit-union-c-t-l-18-5-7-offline)

## Attribution

This repository is based on the code of [Ditto].
However, I didn't apply any tricks used in Ditto (e.g. knowledge injection, sentence summation, augmentation etc.). I used ditto just fine-tuning the BERT model. Since Ditto assumes the input text is English, it can't be adopted to 4sq competition straightforward, where the input text is multi-lingual. In particular, in some Asian languages (Japanese, Chinese etc.), tokens are not space-separated, but ditto assumes that they are space-separated.

[Ditto]: https://github.com/megagonlabs/ditto