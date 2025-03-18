# Inhibition

## Setting up Transformer and Running Experiment

1. Data

Large corpora of crawled news, collected since 2007. https://data.statmt.org/news-crawl/en/ 

**news.2024.en.shuffled.deduped.gz** data

Stored in local ```./data ``` folder

Processing: 
```bash
python preprocess_data_gz.py
```

It will create txt files for full data, training data (50%), and validation data (50%) in 
```
source_file = Path("./data/news.2024.en.shuffled.deduped.txt")
train_file = Path("./data/news.2024.en.train.txt")
valid_file = Path("./data/news.2024.en.valid.txt")
```

Note that the vocab size of training and validation are the same to ensure the vocab size inputted to the model stays constant.

2. Perplexity Experiment
```bash
python ./tests/test_transformer.py
```