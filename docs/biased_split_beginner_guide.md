---
title: "Biased Split: A Beginner's Guide"
author: "biased-split project guide"
date: "September 2026"
geometry: margin=0.8in
fontsize: 10.5pt
colorlinks: true
linkcolor: blue
urlcolor: blue
---

# 1. What this project is about

A machine-learning model can appear to perform well simply because the test molecules are very similar to molecules it saw during training. That is useful if the future data will also be very similar. It is less useful if the real goal is to predict activity for new chemical territory.

This project studies that distinction by creating train/test splits with a controllable chemical difficulty, or **bias**. Instead of asking only:

> Can the model predict a random test set?

we can ask more informative questions:

- Can it predict molecules involved in activity cliffs?
- Can it predict molecules for which a nearest-neighbour model fails?
- Can it predict molecules that are structurally distant from the training set?
- Can it predict molecules in a selected range of a proxy property such as cLogP?

The project then trains a regression model on each split and measures how performance changes as the intended bias increases.

The central idea is simple:

1. Represent each molecule numerically.
2. Define a chemically meaningful question.
3. Construct a train/test split where more test molecules answer that question.
4. Train a model using the training molecules only.
5. Evaluate on the test molecules.
6. Compare performance across bias levels.

# 2. Quick start

From the repository root, start the app with:

```bash
.venv/bin/python -m streamlit run app/app.py
```

The app opens in a browser. It uses a built-in standardized sample dataset unless a CSV is uploaded.

The current sample dataset is:

```text
data/standardized/target_CHEMBL203-1.IC50.csv
```

The default sample columns used by the app are:

- `standardized_smiles`: the molecular structure in SMILES notation.
- `pchembl_value`: the numerical activity value used as the regression target.

# 3. The data and the prediction problem

## 3.1 SMILES

SMILES is a text representation of a molecular graph. For example, `CCO` represents ethanol. The app converts each SMILES string into an RDKit molecule before calculating fingerprints or molecular properties.

A SMILES string is not used directly as the model input. It is converted into a fingerprint.

## 3.2 Activity

The activity column is the value the model tries to predict. In this project it is normally a pChEMBL-like value. Larger or smaller values have meaning determined by the dataset and assay convention, so always inspect the column and its units before interpreting a result.

## 3.3 Standardization

The notebooks describe a preprocessing workflow that canonicalizes structures, removes failed structures, and removes duplicates. This matters because inconsistent representations can make the same molecule look like different molecules.

# 4. Molecular fingerprints and similarity

## 4.1 ECFP4 fingerprints

The app uses a Morgan fingerprint with radius 2 and 2048 bits. This is commonly called an ECFP4-style fingerprint because a radius of 2 roughly captures circular environments up to diameter 4 bonds.

Each molecule becomes a binary vector:

```text
[0, 1, 0, 0, 1, ...]
```

A bit indicates that a particular local molecular environment was found. The vector is not a complete chemical description, but it is a useful representation for comparing structures.

The current implementation uses RDKit's `MorganGenerator` API.

## 4.2 Tanimoto similarity

For binary fingerprints, Tanimoto similarity is:

$$
T(A,B) = \frac{|A \cap B|}{|A| + |B| - |A \cap B|}
$$

It ranges from 0 to 1:

- 0 means no shared fingerprint bits.
- 1 means identical fingerprint bit sets.

The activity-cliff and kNN methods use similarity thresholds such as 0.7. A pair qualifies as chemically similar when its similarity is at least that threshold.

## 4.3 Tversky similarity

The substructure-distance method uses Tversky similarity with $\alpha=1$ and $\beta=0$. For directed molecules $A$ and $B$:

$$
Tv(A,B) = \frac{|A \cap B|}{|A \cap B| + \alpha|A \setminus B| + \beta|B \setminus A|}
$$

With $\alpha=1$ and $\beta=0$, the score emphasizes whether the features of one molecule are contained in the other. The repository computes both directions and keeps the larger value for the symmetric matrix used by the graph algorithm.

# 5. What "bias" means

The project uses two related quantities.

## 5.1 Intended bias

This is the target fraction used while constructing the split. For example, with a test fraction of 0.20 and intended bias 0.50, the splitter tries to make approximately half of the test set satisfy the chosen chemical question.

The number of target biased test molecules is approximately:

$$
\left\lfloor \text{intended bias} \times \text{test fraction} \times N \right\rfloor
$$

where $N$ is the number of molecules.

## 5.2 Effective bias

This is measured after the split has been created. It is the actual fraction of test molecules that satisfy the chemical question relative to the training set.

Effective bias may differ from intended bias because:

- the dataset may not contain enough qualifying molecules;
- whole scaffolds or connected components must sometimes be kept together;
- test-set size is rounded to an integer;
- candidate molecules may compete for the same training molecules;
- the selected similarity and activity thresholds define the available candidates.

Always report effective bias. It is the property the final split actually has.

# 6. The six available splitting methods

| Method | Chemical question | Main controls | Effective bias means |
|---|---|---|---|
| Activity Cliff | Does a test molecule have a similar training molecule with a large activity difference? | Similarity threshold, activity gap | Fraction of test molecules with at least one cross-partition activity cliff |
| kNN Failure | Would the k nearest qualifying training molecules disagree with the test activity? | Similarity threshold, activity gap, $k$ | Fraction of evaluable test molecules where the kNN prediction fails |
| Substructure Distance | Is the test molecule sufficiently disconnected from the training set? | Tversky threshold | Fraction of test molecules with no sufficiently similar training molecule |
| Proxy Sorted | Is the test molecule inside a selected proxy-property range? | Proxy range | Fraction of test molecules inside the ideal range |
| Murcko Scaffold | Does the test set contain complete scaffold groups? | Test fraction | Not a bias-controlled split; effective bias is 0 in the app |
| Random | Is the test set a random baseline? | Test fraction | Not a bias-controlled split; effective bias is 0 in the app |

## 6.1 Activity Cliff Split

An activity cliff is a pair of structurally similar molecules with a large activity difference. The app uses two conditions:

```text
similarity >= similarity threshold
absolute activity difference >= activity threshold
```

The default values are:

- Similarity threshold: 0.70.
- Activity threshold: 1.00 activity units.
- Test fraction: 0.20.
- Intended bias: 0.50.

The splitter builds candidate cliff edges. It walks through those edges and places one molecule in training and the other in testing. The more highly connected molecule is preferentially placed in training, because it can support several test molecules. Remaining test slots are filled from non-cliff molecules first, then cliff molecules if necessary.

This split asks whether the model can handle local chemical changes that produce large activity changes.

## 6.2 kNN Failure Split

For each molecule, the algorithm finds up to $k$ similar neighbours using the training data. It calculates the neighbours' mean activity and compares that value with the molecule's true activity.

A failure is recorded when:

$$
|y_i - \operatorname{mean}(y_{\text{k nearest training neighbours}})|
\geq \text{activity threshold}
$$

The default values are:

- Similarity threshold: 0.70.
- Activity threshold: 1.00.
- Number of neighbours $k$: 3.
- Test fraction: 0.20.
- Intended bias: 0.50.

Important limitation: a molecule needs enough qualifying training neighbours to be evaluated. Molecules without enough neighbours are excluded from the effective-bias denominator.

The notebooks note that activity-cliff splitting can be viewed as a special 1-nearest-neighbour disagreement case.

## 6.3 Substructure Distance Split

This method treats molecules as nodes in a similarity graph. An edge is drawn when Tversky similarity is at least the selected threshold. Connected components represent groups of molecules linked by sufficiently strong structural similarity.

The splitter selects components for the test set so that some test molecules are isolated from the training set. It then fills the remaining test slots randomly.

The default similarity threshold is 0.70 and the default test fraction is 0.20.

A test molecule is counted as isolated when its maximum Tversky similarity to every training molecule is below the threshold:

$$
\max_{j \in \text{train}} Tv(i,j) < \text{threshold}
$$

This split is useful for studying extrapolation to chemical regions that are not well represented in training.

## 6.4 Proxy Sorted Split

This method chooses a proxy property and a range considered chemically interesting or challenging. The current app uses RDKit cLogP as the proxy.

Default proxy range:

```text
2.0 <= cLogP <= 3.0
```

The intended bias is the target fraction of test molecules inside that range. Effective bias is the fraction actually inside the range after the test set is completed.

This is not automatically a measure of molecular novelty. It measures enrichment for a selected property range. A proxy should be chosen because it represents a meaningful deployment or scientific question.

## 6.5 Murcko Scaffold Split

A Murcko scaffold is a simplified representation of a molecule's core ring and linker structure. The splitter groups molecules by scaffold and tries to place whole groups into the test set.

This is a useful scaffold generalization baseline, but the test fraction may not be exact because scaffold groups cannot always be divided. It does not have an intended-bias sweep in the app.

## 6.6 Random Split

The random splitter shuffles molecule indices using a fixed random seed and takes the requested fraction as test data. It is a baseline, not a chemically difficult split.

A strong random-split score does not prove that the model will generalize to new scaffolds, activity cliffs, or other difficult regions.

# 7. The model analysis

For every selected split, the app trains one of two regressors using the fingerprints of the training molecules only.

## 7.1 Random Forest

The default Random Forest uses:

- 100 trees.
- Random seed 42.
- All available CPU workers.

A Random Forest is an ensemble of decision trees. It can model nonlinear relationships between fingerprint bits and activity.

## 7.2 Ridge Regression

Ridge Regression uses:

- Regularization strength $\alpha=1.0$.

It is a linear model with L2 regularization. It is a useful simpler comparison against the nonlinear Random Forest.

## 7.3 Metrics

### RMSE

$$
RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}
$$

RMSE penalizes large errors more strongly. Lower is better and the units match the activity value.

### MAE

$$
MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i-\hat{y}_i|
$$

MAE is the average absolute error. Lower is better and it is easier to interpret as a typical error size.

### R-squared

$$
R^2 = 1 - \frac{\sum_i(y_i-\hat{y}_i)^2}{\sum_i(y_i-\bar{y})^2}
$$

Higher is generally better. An $R^2$ below zero means the model performs worse than predicting the test-set mean under this metric. R-squared can be unstable on small test sets, so do not use it alone.

# 8. The bias sweep analysis

The benchmark tab evaluates intended bias values:

```text
0.0, 0.1, 0.2, ..., 1.0
```

For each value, the app:

1. Creates a new train/test split.
2. Measures effective bias.
3. Trains the selected model on the training fingerprints.
4. Predicts the test activities.
5. Records R2, RMSE, MAE, train size, and test size.

The app shows three outputs:

1. **Performance plot:** R2, RMSE, and MAE against effective bias.
2. **Bias calibration plot:** intended bias against effective bias.
3. **Results table:** every sweep point, with the lowest-RMSE row highlighted.

## 8.1 Choosing the best bias level

The current app recommends the row with the lowest RMSE. This is a reasonable primary rule because RMSE is in the activity units and emphasizes large prediction errors.

However, treat the highlighted row as a diagnostic recommendation, not a universal scientific truth. Also inspect:

- whether MAE improves at the same bias;
- whether R-squared is stable;
- whether the effective bias follows the intended bias;
- whether the test set is large enough;
- whether the chosen bias corresponds to the deployment question.

A bias level that gives the best score is not necessarily the most scientifically important level. Sometimes the purpose is to measure degradation under a difficult split, not to optimize the model.

# 9. How to use the Streamlit app

## Step 1: Choose the dataset

Use the built-in sample dataset or upload a CSV. Confirm that the selected columns contain valid SMILES and numeric activity values.

## Step 2: Choose columns

Select:

- the SMILES column;
- the activity column.

## Step 3: Choose a split

Start with Random Split as a baseline. Then compare it with:

1. Activity Cliff Split.
2. kNN Failure Split.
3. Substructure Distance Split.
4. Proxy Sorted Split.
5. Murcko Scaffold Split.

## Step 4: Set parameters

Use the default settings first. Change one parameter at a time so you can understand its effect.

For example:

- Increase the activity gap to make activity cliffs rarer but stronger.
- Increase similarity threshold to require closer structural matches.
- Increase $k$ to make kNN failure depend on a broader local neighbourhood.
- Narrow the cLogP range to make the proxy condition more selective.

## Step 5: Inspect the t-SNE plot

The t-SNE plot is a visual summary of fingerprint-space relationships. Hover over a point to see its SMILES, partition, activity, and molecule image.

Important: t-SNE is for visualization, not a formal measure of distance or model performance. It can distort global geometry.

## Step 6: Inspect the distribution and prediction plots

The activity distribution shows whether train and test activity ranges differ. The prediction plot compares true test activity with predicted test activity.

Points close to the diagonal are better predictions. Always check whether a small number of test molecules is making the plot look better or worse than it should.

## Step 7: Run the bias sweep

Run the sweep only for a bias-controlled splitter. Review the effective-bias calibration plot first. If intended and effective bias diverge strongly, the dataset or thresholds may not support the requested bias range.

Then use the table and the highlighted lowest-RMSE row as a starting point for interpretation.

# 10. A worked interpretation pattern

Suppose the random split gives RMSE 0.45 and an activity-cliff split at effective bias 0.80 gives RMSE 0.90. A careful interpretation is:

> The model performs well when test molecules are sampled randomly, but its error increases when the test set contains many molecules with close structural neighbours and large activity differences. This suggests the model has difficulty with local activity cliffs.

Do not conclude that the model is universally poor. The result is conditional on:

- the fingerprint;
- the activity threshold;
- the similarity threshold;
- the test fraction;
- the model type;
- the dataset quality.

# 11. Important limitations

## 11.1 One split is not enough

The app uses a fixed random seed for reproducibility. A single split can still be unrepresentative. For research conclusions, repeat the split with several seeds and report mean and variation.

## 11.2 Small test sets are unstable

With a 20% test fraction, a small dataset may contain only a few test molecules. Metrics can change substantially when one molecule moves between train and test.

## 11.3 Similarity depends on the representation

ECFP4 and its thresholds define what "similar" means. A different fingerprint or descriptor could produce different cliffs, neighbours, and components.

## 11.4 Activity measurements contain noise

Assay source, protocol, units, and curation affect the activity values. A large activity difference may partly reflect measurement noise rather than a real chemical effect.

## 11.5 Bias is not automatically bad

Bias here is a deliberate experimental design variable. It is useful because it lets us measure a model under specific chemical conditions. The goal is not always to eliminate bias; the goal is to understand it.

## 11.6 Data leakage must be avoided

The model must be trained only on training molecules. Do not standardize or select features using information that would only be available from the test set. The split question itself may use the full dataset to construct the partition, but model fitting and evaluation must remain separated.

# 12. Recommended analysis checklist

Before trusting a result, record:

- dataset name and version;
- SMILES standardization procedure;
- activity column and units;
- number of molecules after cleaning;
- fingerprint type, radius, and bit size;
- splitter name;
- test fraction;
- similarity and activity thresholds;
- intended and effective bias;
- proxy range, if used;
- model type and hyperparameters;
- random seed;
- RMSE, MAE, and R-squared;
- train and test sizes;
- whether the result was a single split or repeated experiment.

Compare at least one chemically meaningful split with a random baseline. Report effective bias rather than intended bias alone.

# 13. Where to find the implementation

| File | Purpose |
|---|---|
| `app/app.py` | Streamlit interface, controls, plots, and benchmark table |
| `app/descriptors.py` | Morgan fingerprints, similarity matrices, t-SNE, molecule images |
| `app/splitters.py` | Activity-cliff, kNN, substructure, proxy, scaffold, and random splitting logic |
| `app/ml_models.py` | Regressors, metrics, and bias-sweep execution |
| `biased_split/molecularnetwork.py` | Molecular-network helpers and RDKit bit-vector similarity |
| `notebooks/00_Data_Source_and_Standardize.ipynb` | Data source and standardization workflow |
| `notebooks/01_Molecular_Network.ipynb` | Similarity network concepts |
| `notebooks/02_Activity_Cliff_Split.ipynb` | Activity-cliff method development |
| `notebooks/03_kNN_Failure_Split.ipynb` | kNN failure method development |
| `notebooks/04_Substructure_Distance_Split.ipynb` | Tversky and substructure-distance method |
| `notebooks/05_Proxy_Sorted_Split.ipynb` | Proxy-property method |

# 14. Final mental model

Think of the project as a controlled stress test for molecular machine learning.

- The **fingerprint** defines how chemical resemblance is measured.
- The **splitter** defines what kind of difficult test molecule is selected.
- The **intended bias** is the construction target.
- The **effective bias** is what the final split actually contains.
- The **model metrics** quantify how well the model handles that test condition.
- The **bias sweep** shows how performance changes as the test set becomes more concentrated in the selected difficult region.

The most useful conclusion is usually not "the model has an RMSE of X." It is:

> Under which chemically meaningful conditions does the model succeed or fail, and how reliably can we measure that difference?
