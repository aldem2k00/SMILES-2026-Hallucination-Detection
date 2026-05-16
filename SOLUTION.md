1) **Reproducibility instructions**

Same as original repo:
```
git clone https://github.com/aldem2k00/SMILES-2026-Hallucination-Detection.git
cd SMILES-HALLUCINATION-DETECTION

python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate.bat     # Windows

pip install -r requirements.txt
python solution.py
```

2) **Final solution description**

I have modified the three files:
- `aggregation.py`
- `probe.py`
- `splitting.py`

I have also changed package versions in `requirements.txt` to ensure compatibility. However, I expect the code to work with the original version of the `requirements.txt`.

I have also changed `.gitignore` to include `results.json` to the repository.

The feature extraction procedure (`aggregation.py`) involves these steps:

1) The presence of the special tokens `<|im_start|>assistant` and `<|endoftext|>` is identified.
The MAX_LENGTH constant defined in `model.py` (which it is not allowed to change) restricts the maximum length of the processed sequence to 512 tokens. This results in the truncation of some of the examples from the dataset. In the training data, 152 examples out of 689 exceed 512 tokens, and in 7 cases, the truncated sequence does not even contain the full prompt.
The positions of the special tokens are identified by comparing the input embeddings found at index 0 along the `n_layers` dimension to the weights of the Embedding layer.
This results in 2 features. Both are binary (one or zero). The first one is positive if and only if the prompt exceeds 512 tokens, and the second one is positive if and only if the whole sequence (prompt with response) exceeds 512 tokens.
The position where the LLM-generated response starts is used in step 2 to cut the response out of the whole sequence.
2) At every position in the response part of the sequence, the entropy of the next token distribution is calculated. `model.lm_head` submodule is applied to the hidden state in order to calculate the logits, which are then softmaxed, and the entropy is computed. A few sample statistics are computed over all the entropies which are used as a part of the feature vector (minimum value, mean value, maximum value, 25th percentile, 50th percentile, 75thpercentile, standard deviation, log-length of the response) along with the entropy of the first generated token. If all of the response is lost due to truncation, these features are just all set to zero.
This results in 9 features.
3) The Hidden Score method (https://neurips.cc/virtual/2024/poster/95584) is used to extract features from the hidden states at all layers of the LLM.
This results in 24 features, one feature for every layer of the transformer.
4) The rest 896 features are obtained by the default method already implemented in the original repository (last hidden state at the end of the sequence).

In total, 2 + 9 + 24 + 896 = 931 features are extracted.

The probe classifier is a neural network (`probe.py`). 

The 931-dimensional feature vector is splitted into four parts originating from the different feature extraction methods described above. For the three parts (generation entropy statistics, the Hidden Score, and the default features) separate learnable linear transformations are applied. These three vectors are thus mapped to the 44-dimensional space. The weighted sum of the three resulting vectors is then computed. The coefficients of the weighted sum depend on the two binary features related to the sequence truncation.
After the ELU activation, those two binary features are concatenated to the 44-dimensional hidden state, and another linear transformation of the resulting 46-dimensional vector results in the final logits.

For training, I use Adam optimizer with constant learning rate of 1.55556e-4 and train without splitting into batches for 235 epochs.

For evaluation, I employ the StratifiedShuffleSplit from the sklearn module. The train:val:test ratio is 60:20:20. All examples from the data.csv are later used to train the probe for predicting the class labels on the test set.

The method involving generation entropies I have invented out of intuition. Then I have checked if anything had been implemented by the community to really use all hidden states for the prediction, and I found the Hidden Score method which I immediately implemented. Those two methods contributed the most to the average test AUROC.

The hyperparameters hardcoded into the `Probe` were found using Optuna.

The average test AUROC over all folds is **77.35 %.**

3) **Experiments and failed attempts**

I have tried to use every one of the methods I described above separately, and they all proved better than the baseline dummy classifier. However, the combination of the three methods performed best.

I have also tried to reduce the dimensionality of the hidden states using PCA, but those compressed repserentations were found to be useless in training.

I have also tried decoding the whole generated sequence from the hidden states in order to analyze the probabilities of actually chosen tokens, but could not use those probabilities in a way that improved the metric.