# Uncertainty Quantification in Natural Language Explanations of Language Models

This is the source code for all the experiments pertaining to the paper "Uncertainty Quantification in Natural Language Explanations of Language Models" under review at AISTATS 2024.

As mentioned in the paper, We conduct experiments using three math word problem and two commonsense reasoning benchmark datasets. i) the **GSM8K** dataset that comprises several math word problems, ii) the **SVAMP** dataset contains math word problems with varying structures, iii) the **ASDiv** dataset consisting of diverse math word problems, iv) the **StrategyQA** requires a language model to deduce a multi-step reasoning strategy to answer questions and v) the **Sports Understanding** dataset, which is a specialized evaluation set from the BIG-bench that involves determining whether a sentence relating to sports is plausible or implausible.

The processed files are available in `data` directory.



## Steps to reproduce results

### 1. Dataset EDA

The notebook 'EDA.ipynb' loads datasets and does pre-processing of questions. Moreover, also prepends the handcrafted prompts to the question to generate natural language explanations. The final output parquet files are written to `data` directory. The columns are as follows.

`experiment_1_question` - Verbalized Chain Of Thought question
`experiment_2_question` - Feature Importance Explanation question
`experiment_3_question` - Verbalized Feature Importance question
`experiment_4_question` - Paraphrased questions for Sample Probing CoT Explanations
`experiment_5_question` - Paraphrased questions for Sample Probing Feature Importance Explanations
`experiment_8_question` - Model Probing CoT Explanation question
`experiment_9_question` - Model Probing Feature Importance Explanation question

### 2. Predictions with GPT models

The script `uncertainty-all.py` takes a dataset file path as argument, and gets responses from GPT models using OpenAI's API's. Make sure you update the api key and model name in the script accordingly.

**Example**

```bash
python uncertainty-all.py data/gsm8k_100/input.parquet
```

### 3. Computing Uncertainty

For math word problem datasets, run the notebook "Math Word Problem Analysis.ipynb" to compute verbalized, sample probing and model probing uncertainty scores of explanations.

For commonsense reasoning datasets, run the notebook "Commonsense Reasoning Analysis.ipynb" to compute verbalized, sample probing and model probing uncertainty scores of explanations.

### 4. Computing Faithfulness

For math word problem datasets, run the notebook "Math Word Problem Analysis.ipynb" to compute faithfulness of token importance explanations, and CoT dataset faithfulness.

For commonsense reasoning datasets, run the notebook "Commonsense Reasoning Analysis.ipynb" to compute faithfulness of token importance explanations, and CoT dataset faithfulness.

### 5. Plots

The notebook `Plots.ipynb` has plotting code for the results shown in the paper.


## Results

The final processed files for all 5 datasets, and 2 models, can be accessed here.

```Python
import pandas as pd

data_all = {
    "GSM8K": {
        "text-davinci-003": pd.read_parquet("data/gsm8k_100/faithfulness-scores-text-davinci-003.parquet"),
        "gpt-3.5-turbo": pd.read_parquet("data/gsm8k_100/faithfulness-scores-gpt-3.5-turbo.parquet"),
    },
    "SVAMP": {
        "text-davinci-003": pd.read_parquet("data/svamp_100/faithfulness-scores-text-davinci-003.parquet"),
        "gpt-3.5-turbo": pd.read_parquet("data/svamp_100/faithfulness-scores-gpt-3.5-turbo.parquet"),
    },
    "ASDiv": {
        "text-davinci-003": pd.read_parquet("data/asdiv_100/faithfulness-scores-text-davinci-003.parquet"),
        "gpt-3.5-turbo": pd.read_parquet("data/asdiv_100/faithfulness-scores-gpt-3.5-turbo.parquet"),
    },
    "StrategyQA": {
        "text-davinci-003": pd.read_parquet("data/strategyqa_100/faithfulness-scores-text-davinci-003.parquet"),
        "gpt-3.5-turbo": pd.read_parquet("data/strategyqa_100/faithfulness-scores-gpt-3.5-turbo.parquet"),
    },
    "Sports Understanding": {
        "text-davinci-003": pd.read_parquet("data/sportsunderstanding_100/faithfulness-scores-text-davinci-003.parquet"),
        "gpt-3.5-turbo": pd.read_parquet("data/sportsunderstanding_100/faithfulness-scores-gpt-3.5-turbo.parquet"),
    },
}
```