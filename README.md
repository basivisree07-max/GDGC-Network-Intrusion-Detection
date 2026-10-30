# Network Intrusion Detection using Machine Learning

**Competition:** Network Intrusion Detection — ML Challenge (GDG on Campus AI/ML Volunteer Recruitment Challenge)
**Platform:** Kaggle Notebooks (CPU only)
**Author:** `<Your Name>`

A multiclass classification project that labels network-traffic flows as **Benign**, **Availability**, **Reconnaissance** or **Spoofing**. The solution is a Scikit-learn `Pipeline` that combines one-hot encoding of the two categorical features with a `RandomForestClassifier`. It reaches a **Macro F1 of about 0.9881** on a stratified 20% validation split. The notebook also shows that the test data differs strongly from the training data, so that validation score should not be read as an estimate of performance on the hidden test set (see [Section 10](#10-observations-and-limitations)).

---

## Table of Contents

1. [Project Title and Description](#1-project-title-and-description)
2. [Problem Statement](#2-problem-statement)
3. [Dataset Description](#3-dataset-description)
4. [Exploratory Data Analysis (EDA)](#4-exploratory-data-analysis-eda)
5. [Data Preprocessing](#5-data-preprocessing)
6. [Train-Validation Split](#6-train-validation-split)
7. [Model Selection and Implementation](#7-model-selection-and-implementation)
8. [Model Evaluation](#8-model-evaluation)
9. [Test Predictions and Submission](#9-test-predictions-and-submission)
10. [Observations and Limitations](#10-observations-and-limitations)
11. [Future Improvements (Proposed Work)](#11-future-improvements-proposed-work)
12. [Conclusion](#12-conclusion)
13. [Technologies Used](#13-technologies-used)
14. [Project Files and How to Reproduce](#14-project-files-and-how-to-reproduce)

---

## 1. Project Title and Description

**Network Intrusion Detection using Machine Learning**

A Network Intrusion Detection System (NIDS) monitors network traffic and flags activity that looks malicious. This project builds a supervised machine-learning model for that task. Each row in the dataset describes one network flow using 82 numerical and categorical features (packet counts, header sizes, TCP flag counts, timing statistics, ports, protocol and service). The model predicts which of four traffic categories the flow belongs to.

The notebook covers:

- exploratory analysis of the training and test files,
- a preprocessing step built with `ColumnTransformer`,
- a stratified train/validation split,
- a Random Forest baseline wrapped in a `Pipeline`,
- evaluation with Macro F1, a classification report and a confusion matrix,
- generation of `submission.csv` for the test set,
- a comparison of training and test feature distributions.

---

## 2. Problem Statement

**Objective.** Given the statistical description of a network flow, decide whether it is normal traffic or one of several kinds of malicious traffic.

**Target classes.** The classes below are described in general terms. The notebook itself does not contain a data dictionary, so these are general definitions, not statements about how the data was generated.

| Class | General meaning |
|---|---|
| **Benign** | Normal, non-malicious traffic. |
| **Availability** | Attacks that try to make a service unavailable or degraded (for example, flooding-style attacks). |
| **Reconnaissance** | Probing or scanning activity used to discover hosts, open ports or services before an attack. |
| **Spoofing** | Traffic in which the sender falsifies its identity (for example, forged addresses) to impersonate another party. |

**Why multiclass classification?**
Each flow belongs to exactly one of four categories, so this is a single-label, four-class problem. A simple "malicious vs. benign" model would hide useful information. A security team would respond differently to a denial-of-service style attack than to port scanning or identity spoofing. The model therefore has to separate all four categories.

---

## 3. Dataset Description

The data comes from the competition and is read from three files: `train.csv`, `test.csv` and `sample_submission.csv`.

| File | Rows | Columns | Notes |
|---|---|---|---|
| `train.csv` | 84,713 | 84 | `id` + 82 input features + `Target` |
| `test.csv` | 33,161 | 83 | `id` + 82 input features (no `Target`) |
| `sample_submission.csv` | — | 2 | `id`, `Target` |

**Column roles**

- **`id`**: a unique identifier for each record (values look like `train_150a9d…` and `test_b1eb82…`). It is not a traffic measurement. It is only used to link each prediction to its test row, and it is removed from the model inputs.
- **`Target`** (training file only): the class label to predict. It is the dependent variable `y`.
- **82 input features**: the remaining columns, used as the independent variables `X`.

**Feature types** (from `train.dtypes`: 80 `float64` columns and 4 `object` columns, where the four object columns are `id`, `proto`, `service` and `Target`)

- **Categorical features (2):**
  - `proto`: transport protocol (`tcp`, `udp`, `icmp` in the training data).
  - `service`: application-level service (`-`, `dns`, `mqtt`, `ssl`, `http`, `ntp`, `irc`, `dhcp`, `radius` in the training data). The value `-` appears to mean that no specific service was identified.
- **Numerical features (80):** flow-level statistics. Judging by the column names, they include source and destination ports (`id.orig_p`, `id.resp_p`), flow duration, forward and backward packet counts, packets per second, header sizes, TCP flag counts (FIN, SYN, RST, PSH, ACK, URG, CWR, ECE), payload-size statistics, inter-arrival-time statistics (`*_iat.*`), subflow and bulk-transfer statistics, active/idle time statistics, and initial/last TCP window sizes. This grouping is inferred from the column names, because the notebook does not include official feature descriptions.

---

## 4. Exploratory Data Analysis (EDA)

All findings below come from the printed outputs of the notebook. The notebook uses printed summaries and tables only and contains no plots.

### 4.1 Dataset dimensions

- Training data: **(84,713, 84)**
- Test data: **(33,161, 83)**
- After dropping `id` (and `Target` for training), both feature sets have **82 columns**, and their column names and order match exactly (`Matching columns: True`).

### 4.2 Target class distribution

| Class | Count | Percentage |
|---|---:|---:|
| Availability | 80,072 | 94.52% |
| Spoofing | 2,323 | 2.74% |
| Benign | 1,583 | 1.87% |
| Reconnaissance | 735 | 0.87% |

### 4.3 Class imbalance

The dataset is **heavily imbalanced**. Availability makes up about 94.5% of the training data, while Reconnaissance makes up under 1%. The ratio between the largest and smallest class is roughly 109 to 1 (80,072 / 735). This has two consequences for the rest of the project:

- **Accuracy is misleading.** A model that always predicted "Availability" would score about 94.5% accuracy on data with this class mix (an illustrative calculation, not a model from the notebook) while detecting no other attack. This is why Macro F1 is used for evaluation (Section 8).
- **The split and the model must account for imbalance.** This motivates the stratified split (Section 6) and the class weights in the Random Forest (Section 7).

### 4.4 Categorical feature analysis

**`proto` (training data)**

| Value | Count |
|---|---:|
| tcp | 82,707 |
| udp | 1,974 |
| icmp | 32 |

**`service` (training data)**

| Value | Count |
|---|---:|
| `-` | 80,790 |
| dns | 1,215 |
| mqtt | 951 |
| ssl | 921 |
| http | 666 |
| ntp | 115 |
| irc | 28 |
| dhcp | 25 |
| radius | 2 |

Both features are dominated by a single value (`tcp`, `-`). Some categories, such as `radius` with 2 records and `icmp` with 32, are very rare.

### 4.5 Missing values

There are **no missing values** in either file (total missing values: 0 in train and 0 in test). No imputation was needed.

### 4.6 Duplicate values

`train.duplicated().sum()` returned **0**. One caveat: this check was run on all columns, including `id`. If `id` is unique per record (as the identifiers suggest), no row could be a duplicate in this check. A duplicate check on the 82 feature columns alone was **not** performed in the notebook.

### 4.7 Infinite values

There are **no infinite values** in the numeric columns of the training data or the test data (0 in each). This matters because some columns are rates (for example, packets per second), which can overflow to infinity when a duration is zero. That did not happen here.

### 4.8 Other observations

- **Data types:** 80 `float64` columns and 4 `object` columns (`id`, `proto`, `service`, `Target`).
- **Feature compatibility:** the test file has the same 82 feature columns as the training file, in the same order.
- **Submission format:** `sample_submission.csv` has two columns, `id` and `Target`.
- **Train vs. test comparison:** the notebook compares training and test distributions in detail. Because this finding is central to the project, it is covered in [Section 10](#10-observations-and-limitations).

---

## 5. Data Preprocessing

### 5.1 Separating X and y

- `y = train['Target']` is the dependent variable (the label).
- `X = train.drop(columns=['id', 'Target'])` holds the independent variables (82 features).

### 5.2 Removing `id` and `Target`

- **`id` is removed** because it is only an identifier. It carries no information about the traffic and could let the model memorize meaningless values.
- **`Target` is removed from `X`** because it is the answer. Leaving it in would be target leakage.
- **For the test set**, only `id` is dropped (`test.drop(columns=['id'])`), because the test file has no `Target` column. The notebook verifies that the training and test feature columns are identical.

### 5.3 Identifying categorical and numerical features

```python
categorical_features = ['proto', 'service']
numerical_features = [col for col in X.columns if col not in categorical_features]
```

This gives **2 categorical features** and **80 numerical features**.

### 5.4 One-Hot Encoding

Machine-learning models need numbers, but `proto` and `service` contain text labels. `OneHotEncoder` replaces each categorical column with one binary column per category. For example, `proto` becomes separate indicator columns for `tcp`, `udp` and `icmp`. One-hot encoding is used rather than converting categories to 1, 2, 3, because integer codes would suggest a numeric order between categories that does not exist.

### 5.5 Why `handle_unknown='ignore'`

If the encoder meets a category at prediction time that it never saw during fitting, it raises an error by default. With `handle_unknown='ignore'`, an unseen category is encoded as all zeros instead, and the pipeline still runs. This is a safeguard. Test data can contain categories that are absent from the data the encoder was fitted on, and rare categories such as `radius` (2 records) could even be absent from the training portion of a random split. From the printed output, the test file's `proto` and `service` values (tcp/udp; `-`, dns, mqtt, http, ssl) all occur in the training file, so the notebook does not show an unseen category in the test data.

### 5.6 `ColumnTransformer`

`ColumnTransformer` applies a different transformation to each group of columns and joins the results into a single feature matrix:

```python
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', 'passthrough', numerical_features)
    ]
)
```

- Categorical columns go through the one-hot encoder.
- Numerical columns go through unchanged.

### 5.7 Why numerical features were passed through without scaling

The numerical features are passed through (`'passthrough'`) with no standardization or normalization. The reason is the model choice. Random Forests are built from decision trees, and a tree only asks questions of the form "is feature *j* below threshold *t*?". Rescaling a feature changes the threshold values but not the order of the samples, so the resulting splits are essentially the same. Scaling matters for distance-based or gradient-based models (k-NN, SVM, logistic regression, neural networks), not for tree ensembles. Skipping it also keeps the pipeline simpler. The notebook did not run an experiment comparing scaled and unscaled inputs, so this is a design rationale and not a measured result.

---

## 6. Train-Validation Split

```python
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

- **80:20 split.** 80% of the labelled data is used to train the model, and 20% is held back to check how well the model performs on data it has not seen. The test file has no labels, so a validation split is the only way to measure performance inside the notebook.
- **`train_test_split`** is Scikit-learn's standard function for shuffling and splitting data.
- **`random_state=42`** fixes the random shuffle so the same split is produced every time the notebook runs. This makes the results reproducible and comparable between runs. The value 42 is a common convention, and the notebook did not test other seeds.
- **`stratify=y`** keeps the class proportions the same in both parts. This is essential here. With a plain random split, a class as small as Reconnaissance (0.87%) could end up under-represented in one part by chance, which would make training or evaluation unreliable.

**Resulting sizes**

| Set | Shape |
|---|---|
| Training | (67,770, 82) |
| Validation | (16,943, 82) |

**Class distributions**

| Class | Training | Validation |
|---|---:|---:|
| Availability | 64,057 | 16,015 |
| Spoofing | 1,858 | 465 |
| Benign | 1,267 | 316 |
| Reconnaissance | 588 | 147 |

The proportions in both parts match the full training set (for example, Reconnaissance is about 0.87% in each), which confirms that stratification worked.

---

## 7. Model Selection and Implementation

### 7.1 What is a Random Forest?

A **Random Forest** is an ensemble of many decision trees. A single decision tree classifies a sample by asking a series of yes/no questions about its features (for example, "is `flow_duration` below some value?"). A single tree tends to overfit the training data. A Random Forest reduces this by:

1. training each tree on a different random sample of the training data (bootstrap sampling),
2. letting each tree consider only a random subset of features at each split, so the trees differ from one another,
3. combining the predictions of all trees by majority vote.

Averaging many different trees gives a more stable model than any one tree.

### 7.2 Why it was chosen as the baseline

Random Forest is a reasonable first model for this kind of data because it:

- works directly on tabular data with many numerical features,
- needs no feature scaling (Section 5.7),
- can model non-linear relationships and interactions between features,
- needs little tuning to give a strong first result,
- supports class weighting for imbalanced data.

It serves as a baseline, meaning a solid reference point against which other models could later be compared. The notebook does not compare it with other models.

### 7.3 Parameters used

```python
RandomForestClassifier(
    n_estimators=200,
    class_weight='balanced_subsample',
    random_state=42,
    n_jobs=-1,
    min_samples_leaf=2
)
```

| Parameter | Value | Purpose |
|---|---|---|
| `n_estimators` | `200` | Number of trees in the forest. More trees generally give more stable predictions at the cost of training time. |
| `class_weight` | `'balanced_subsample'` | Gives each class a weight inversely proportional to its frequency, recomputed for the bootstrap sample of each tree. Errors on rare classes (Reconnaissance, Benign, Spoofing) then count more than errors on Availability, which counters the imbalance from Section 4.3. |
| `random_state` | `42` | Fixes the randomness (bootstrap sampling, feature selection) so training is reproducible. |
| `n_jobs` | `-1` | Uses all available CPU cores to train trees in parallel. |
| `min_samples_leaf` | `2` | Every leaf must contain at least 2 training samples. This stops trees from growing leaves for single samples, which acts as light regularization against overfitting. |

These values were set directly in the notebook. No hyperparameter search or comparison of alternative values was carried out, so they should be treated as reasonable defaults, not tuned optima.

### 7.4 Why a Scikit-learn `Pipeline`

```python
rf_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(...))
])
rf_model.fit(X_train, y_train)
```

The pipeline chains the preprocessing step and the classifier into one object. This has several practical benefits:

- **No leakage from validation data.** When `.fit()` is called, the encoder learns its categories from the training split only. Validation and test data are only transformed, never used for fitting.
- **Consistent preprocessing.** `predict()` automatically applies the same encoding to new data. There is no risk of forgetting a step or applying a different one at prediction time.
- **Cleaner code.** One `fit` call and one `predict` call handle everything.

The model was trained on the 80% training split only. It was not retrained on the full labelled data before predicting the test set.

---

## 8. Model Evaluation

### 8.1 Metrics

For each class:

- **Precision** = TP / (TP + FP). Of the flows the model labelled as this class, the fraction that truly belong to it. Low precision means many false alarms.
- **Recall** = TP / (TP + FN). Of the flows that truly belong to this class, the fraction the model found. Low recall means missed detections.
- **F1 score** = the harmonic mean of precision and recall. It is high only when both are high.

**Macro-averaged F1** computes the F1 score separately for each of the four classes and takes the plain (unweighted) average. Every class therefore counts equally, whatever its size.

**Why Macro F1 suits this competition.** Because of the imbalance (Section 4.3), accuracy and even weighted averages are dominated by the Availability class. A model could be poor on Reconnaissance (735 training rows) and still look excellent overall. Macro F1 gives Reconnaissance the same weight as Availability, so it rewards a model only if it handles *every* attack type well. In intrusion detection, missing a rare attack type is exactly the failure that matters.

### 8.2 Result on the validation set (16,943 samples)

**Macro F1 Score: `0.9881316637189201` (approximately 0.9881)**

**Classification report (from the notebook output)**

```
                precision    recall  f1-score   support

  Availability       1.00      1.00      1.00     16015
        Benign       0.98      0.99      0.99       316
Reconnaissance       0.96      0.99      0.98       147
      Spoofing       0.99      0.99      0.99       465

      accuracy                           1.00     16943
     macro avg       0.98      0.99      0.99     16943
  weighted avg       1.00      1.00      1.00     16943
```

**Confusion matrix (from the notebook output)**

Rows are the true classes and columns are the predicted classes, in the order Availability, Benign, Reconnaissance, Spoofing (the order of the classification report).

| True \ Predicted | Availability | Benign | Reconnaissance | Spoofing |
|---|---:|---:|---:|---:|
| **Availability** | 16010 | 0 | 5 | 0 |
| **Benign** | 0 | 312 | 0 | 4 |
| **Reconnaissance** | 0 | 1 | 146 | 0 |
| **Spoofing** | 0 | 4 | 1 | 460 |

### 8.3 Reading the results

Reading the confusion matrix directly:

- The diagonal holds nearly all samples. Only **15 of 16,943** validation samples were misclassified (5 + 4 + 1 + 4 + 1).
- The few errors are spread over a small number of class pairs. Benign and Spoofing are confused with each other in both directions (4 + 4). Five Availability flows were predicted as Reconnaissance, and one Spoofing flow was predicted as Reconnaissance.
- **Reconnaissance has the lowest precision (0.96).** The reason is visible in the matrix. Of the 152 flows predicted as Reconnaissance, 146 were correct, and 5 of the 6 mistakes were Availability flows. Because Reconnaissance is such a small class, a handful of false positives from the huge Availability class lowers its precision noticeably.
- Accuracy and the weighted average round to 1.00 because Availability dominates the sample count. This is the reason Macro F1 is the more informative number.

This score measures performance on a random 20% split of the *training* data. How far it carries over to the test data is discussed in Section 10.

---

## 9. Test Predictions and Submission

1. **Prepare the test features.** `X_test = test.drop(columns=['id'])`, which keeps the same 82 columns the model was trained on.
2. **Predict.** `test_predictions = rf_model.predict(X_test)`. The pipeline applies the same one-hot encoding learned during training and then the Random Forest.
3. **Preserve the original IDs.** The submission is built from `test['id']`, taken straight from the original test file. Predictions are in the same row order as `X_test`, so each prediction stays aligned with its `id`.
4. **Build the required columns.**
   ```python
   submission = pd.DataFrame({'id': test['id'], 'Target': test_predictions})
   ```
5. **Export.** `submission.to_csv('/kaggle/working/submission.csv', index=False)`.

**Checks performed in the notebook**

- The submission shape is **(33,161, 2)**, so it contains one row for each of the 33,161 test records, with the columns `id` and `Target`.
- The file exists in `/kaggle/working/` (`os.path.exists` returned `True`).
- The notebook did not explicitly compare the submission IDs against `sample_submission.csv` (beyond comparing column names) or check them for duplicates.

**Predicted class distribution on the test set**

| Predicted class | Count | Share of test set |
|---|---:|---:|
| Reconnaissance | 17,812 | 53.71% |
| Benign | 10,576 | 31.89% |
| Spoofing | 3,541 | 10.68% |
| Availability | 1,232 | 3.72% |

These are the model's predictions, not ground truth. The test labels are hidden. They are discussed further in Section 10.

---

## 10. Observations and Limitations

The competition description warns that the public and private leaderboard datasets contain different reserved network-traffic behavior profiles. This section separates **what the notebook established** from **possible explanations** that the notebook did not test.

### 10.1 What is distribution shift?

**Distribution shift** (also called dataset shift) means the data a model sees at prediction time is statistically different from the data it was trained on. It can mean that feature values, category frequencies, class proportions, or the relationship between features and labels have changed. Machine-learning models are trained on the assumption that new data resembles the training data. When that assumption fails, performance can drop even if the model looked excellent during development.

### 10.2 Findings established by the notebook

These are direct results of the code and outputs in the notebook.

**(a) Protocol mix differs sharply (`proto`, share of rows)**

| Protocol | Train | Test |
|---|---:|---:|
| tcp | 97.63% | 68.92% |
| udp | 2.33% | 31.08% |
| icmp | 0.04% | not present |

**(b) Service mix differs sharply (`service`, share of rows)**

| Service | Train | Test |
|---|---:|---:|
| `-` | 95.37% | 52.49% |
| dns | 1.43% | 24.78% |
| mqtt | 1.12% | 9.59% |
| http | 0.79% | 7.91% |
| ssl | 1.09% | 5.23% |
| ntp, irc, dhcp, radius | 0.00% to 0.14% each | not present |

The test set contains only 5 of the 9 service values seen in training, and DNS traffic is about 17 times more common in the test set (24.78% vs. 1.43%).

**(c) Several numerical features have very different averages.** The notebook compared the training and test mean of every numeric feature, scaled by the training standard deviation, and listed the top 15:

| Feature | Train mean | Test mean |
|---|---:|---:|
| `flow_pkts_payload.min` | 1.05 | 44.84 |
| `down_up_ratio` | 0.98 | 0.53 |
| `flow_RST_flag_count` | 0.966 | 0.367 |
| `fwd_pkts_payload.min` | 113.90 | 49.30 |
| `id.resp_p` | 99.35 | 3,481.43 |
| `flow_FIN_flag_count` | 0.030 | 0.338 |
| `fwd_init_window_size` | 2,279.84 | 16,340.50 |
| `payload_bytes_per_second` | 5.69 × 10⁷ | 6.22 × 10⁴ |

`fwd_URG_flag_count` tops the notebook's ranking with an enormous score (about 6 × 10⁸). That number is a side effect of the formula, not a meaningful size. The feature is always 0 in the training data (mean 0, standard deviation 0), so the score divides by almost nothing. The real finding is simple: the URG flag never appears in the training data, but it has a non-zero mean (0.0605) in the test data.

Limits of this comparison: it compares means only, and it was not visualized, so it does not show the full shape of each distribution.

**(d) The predictions on the test set look very different from the training labels.** The training data is 94.52% Availability. The model's test predictions are only 3.72% Availability and 53.71% Reconnaissance (Section 9). This is a fact about the model's output. It does not prove the predictions are right or wrong, because the test labels are unknown.

### 10.3 Why a high validation score may not translate to the leaderboard

The validation score in Section 8 comes from a **random split of the training data**. Validation rows and training rows were drawn from the same pool, so they share the same protocol mix, service mix and traffic behavior. The score therefore answers the question "how well does the model do on data that looks like the training data?" The competition test set is drawn from a different mix (Section 10.2). A Macro F1 of 0.9881 on a random split can therefore overstate the performance on the test data. The notebook did not measure the actual test performance, so how large the gap is cannot be determined from the notebook.

**Generalization**, the ability to perform well on new data that differs from the training data, is what the leaderboard actually tests. A model that fits the training distribution very well but relies on patterns specific to it can lose accuracy when those patterns change.

### 10.4 Possible explanations (hypotheses, not verified in the notebook)

The following are plausible reasons the shift could hurt, but the notebook did not test any of them:

- The model may have learned patterns tied to the training traffic profile (for example, that most flows are TCP with `service = '-'`) and may handle UDP/DNS-heavy traffic less reliably, since UDP and DNS are rare in training.
- Port features (`id.orig_p`, `id.resp_p`) are used as ordinary numbers. The average destination port differs greatly between train and test, so the model might lean on specific port values that do not carry over.
- The test set may have a different class balance from the training set, and the model's predictions (Section 10.2d) would be consistent with that. It is equally possible that the model is misclassifying test traffic that looks unfamiliar. Without test labels these two explanations cannot be told apart here.
- Minority classes (Reconnaissance, Benign, Spoofing) have few training examples, so the model may have seen only a narrow range of their behavior.

### 10.5 Other limitations of this implementation

- **Single validation strategy.** Only one random 80:20 split was used. There was no cross-validation and no validation designed to imitate the train-to-test shift.
- **No leaderboard result in the notebook.** The notebook contains no public or private leaderboard score, and none is claimed in this README.
- **No hyperparameter tuning and no model comparison.** Random Forest with the parameters from Section 7.3 is the only model that was trained.
- **Model trained on 80% of the labelled data.** It was not refitted on the full training set before predicting the test set.
- **No feature engineering, feature selection or feature-importance analysis.**
- **No plots.** The EDA is based on printed tables and counts.
- **Duplicate check includes `id`** (Section 4.6), so it does not rule out duplicate feature rows.
- **Test class labels are unknown**, so the predicted distribution in Section 9 cannot be validated.

---

## 11. Future Improvements (Proposed Work)

> **Everything in this section is proposed future work. None of it was implemented in the notebook.**

- **More robust validation.** Validate in a way that mimics the shift the notebook observed. For example, hold out whole groups of traffic (a protocol, service or port range) so that validation data differs from training data, or use stratified k-fold cross-validation and report the spread of scores across folds. This would give a more honest estimate of the leaderboard performance.
- **Other tree-based models.** Compare the baseline with CatBoost, which handles categorical features natively, and with other gradient-boosted tree models such as XGBoost or LightGBM, using the same validation scheme.
- **Feature engineering.** Examples: ratios between forward and backward traffic, log-transforms of heavy-tailed rate features, and treating port numbers as categories or port ranges (well-known, registered, ephemeral) instead of raw numbers.
- **Handling distribution shift.** Examples: remove or coarsen the features that differ most between train and test (such as raw port numbers); use adversarial validation (training a classifier to tell training rows from test rows) to find the features that cause the shift; try re-weighting training rows to look more like the test data.
- **Hyperparameter tuning.** Tune the Random Forest (`n_estimators`, `max_depth`, `min_samples_leaf`, `max_features`) and any boosted models with cross-validation, scored by Macro F1.
- **Improving minority-class generalization.** Try resampling (for example, SMOTE or targeted undersampling of Availability), per-class threshold or probability calibration, and error analysis of the specific flows that are misclassified.
- **Interpretability.** Use feature importances to check which features drive each prediction and whether they are sensible.

---

## 12. Conclusion

This project built a complete multiclass pipeline for network intrusion detection: data inspection, a preprocessing step with one-hot encoding, a stratified split, a class-weighted Random Forest baseline, and a `submission.csv` containing predictions for all 33,161 test records. On a stratified validation set the model reached a **Macro F1 of about 0.9881**, with only 15 of 16,943 samples misclassified.

The exploration also showed the limits of that number. The training data is extremely imbalanced, and the test data differs strongly from it in protocol mix, service mix and several numerical features. A random-split validation score cannot capture this, so the validation result should be read as performance on training-like data and not as a prediction of the leaderboard score. The main lessons were that metrics must match the problem (Macro F1 instead of accuracy), that preprocessing belongs inside a pipeline, and that checking for distribution shift is as important as tuning the model. The natural next steps are the shift-aware validation and model comparisons listed in Section 11.

---

## 13. Technologies Used

Only technologies that appear in the notebook are listed.

| Technology | Use in this project |
|---|---|
| **Python 3** | Programming language |
| **Kaggle Notebooks** | Development and execution environment (CPU only, no GPU) |
| **pandas** | Loading CSV files, inspecting data, building the submission DataFrame |
| **NumPy** | Selecting numeric columns and checking for infinite values |
| **scikit-learn** | `train_test_split`, `ColumnTransformer`, `OneHotEncoder`, `Pipeline`, `RandomForestClassifier`, `f1_score`, `classification_report`, `confusion_matrix` |

The notebook also imports `kagglehub`, from the default Kaggle template, but does not use it.

---

## 14. Project Files and How to Reproduce

### 14.1 Files in this submission

| File | Description |
|---|---|
| `README.md` | This documentation |
| `notebook617ecbe4a5.ipynb` | The Jupyter notebook with all code and outputs |
| `submission.csv` | Predictions for the 33,161 test records (`id`, `Target`) |

### 14.2 Running the notebook on Kaggle

1. **Join the competition.** Open *Network Intrusion Detection — ML Challenge* on Kaggle and accept the rules so that the data is available to your account.
2. **Create a notebook.** Go to *Code → New Notebook*. Alternatively, open the competition page and choose *New Notebook*, which attaches the competition data automatically.
3. **Add the data.** If the data is not attached, use *Add Input* in the right-hand panel and add the competition dataset. The notebook expects the files at:
   ```
   /kaggle/input/competitions/network-intrusion-detection2-ml-challenge/
       ├── train.csv
       ├── test.csv
       └── sample_submission.csv
   ```
4. **Load the code.** Use *File → Import Notebook* to upload the `.ipynb` file, or copy the cells into the new notebook.
5. **Check the paths.** The first cell prints every file under `/kaggle/input`. If your paths differ from the ones above, update the three `pd.read_csv(...)` calls in the second cell.
6. **Settings.** The original run used no accelerator (CPU only) with internet enabled. No GPU is needed.
7. **Run everything.** Choose *Run All* (or *Save Version → Save & Run All*). The saved run metadata shows the notebook finished in roughly 19 seconds.
8. **Collect the output.** The notebook writes `/kaggle/working/submission.csv`. Download it from the *Output* tab, or submit it directly with *Submit to Competition*.

Because `random_state=42` is set in both the data split and the model, a re-run in the same environment should reproduce the same split and the same results.
