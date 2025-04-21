# Codes
**CODE 0**
```python
# CODE0
# CODE 0
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.style.use(‘winter’)
pd.set_option(‘max_columns’, None)
```
**CODE 1**
```Python
# CODE1
from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values=np.nan, strategy=‘mean’)
imp.fit_transform(X)
```


# General

1. Import Stuff using [CODE0]
2. Train Test split
3. Create and get error of a baseline model (like average or median)
## Data Exploration

1. Open the dataset
2. Data Understanding
	-  DataFrame `.shape`
	- `.head()` and `.tail()`
	- `.dtypes`
	- `.describe()`
	- `.info()`
3. Data Preparation
	- Drop irrelevant columns and rows
		- use `.columns` to obtain list of columns
		- copy and paste column list
		- Remove unnecessary columns
		- Set a variable to be the dataset indexed with the remaining columns, add `.copy()` at the end
		- you can also use `.drop([‘cols you wanna drop’], axis=1, inplace=True)`
	- change necessary column type, using `dataset[column] = pd.to_TheFinalDataType(dataset[column])`
		- You can use `.to_numeric`, `.to_datetime`, etc.
	- Identify duplicate columns
		- use `.duplicated()`
		- use `.loc[.duplicated()]` to look at all duplicate rows
		- use `.duplicated().sum()` to get number of duplicated rows
		- use `.duplicated(subset=[])` to get duplicated of specific columns
		- To remove duplicated use slicing with `~` in front for inverse
		- use `.reset_index(drop=True)` to reset index
	- Rename Columns
		- To rename, `.rename(columns={’old_name’: ‘new_name’}, inplace=True)`
	- Feature Creation
	- Remove or do some other action to `NaN` values
		- Using pandas
			- Identify using `.isna()`
			- Get total using `.isna().sum()`
			- Replace using `.fillna(value or mean, median, mode, etc)`
		- Using Scikit-Learn
			- Replace using mean, most_frequent or constant for strategy, use [CODE1]
4. Feature Understanding (Univariate Analysis) use `ax = ` to save any plot into a matplotlib plot
	- Value Count use `[column].value_counts()` to see number of unique values and their count
	- Histogram, use `.hist(title=“Plot0”)` or `.plot(kind=‘hist’)` on any pd df
	- KDE, use `.plot(kind=‘kde’)`
	- Bar Graph, use `.plot(kind=‘bar’, title=“Plot1”)` or barh instead of bar for horizontal plot
	- Box Plot
5. Feature Relationship
	- Scatter Plot
		- For basic, use `.plot(kind=‘scatter’, x=’column’, y=’column too’)`
		- For Pro, use `sns.scatterplot(data=df, x=‘column’, y=‘column 2’)`, you can use `hue=` property to use other columns as the color 
	- Pair Plot
		- Use `sns.pairplot(data=df, vars=[a very long array of stuff])`
		- Use `sns.pairplot(data=df, vars_x=[…], vars_y=[…])` to get only specific stuff
		- use can also use `hue=` for this also.
	- Correlation, use `.corr()`, you can pass in the type of [pearson, kendall, spearman or callable] 
		- Use `sns.heatmap()` to get better visualization use `annot=True` to see the values inside

## Feature Selection + Data Preparation

1. Use `sklearn.compose.ColumnTransformer` to apply feature selection for individual columns.
2. Use `sklearn.pipeline.make_pipeline` to make pipelines to automate stuff. Column transform can be added to it as a process.
3. Dimensionality Reduction
	- PCA: `sklearn.decomposition.PCA`
	- t-SNE: `sklean.mainfold.TSNE`
	- UMAP: gotta install `umap-learn`, `umap.umap_ as umap` and use as if it was t-SNE or PCA
4. Scaling
	- Normalizer: `sklearn.preprocessing.Normalizer`
	- StandardSalar: `sklearn.preprocessing.StandardScalar`
	- MinMaxSalar: `sklearn.preprocessing.MinMaxScalar`
	- RobustScalar: `sklean.preprocessing.RobustScalar`
5. Feature expansion:
	- PolynomialFeatures: `sklearn.preprocessing.PolynomialFeatures`
6. Use Categorical Encoding
	- One Hot Encoding: `sklearn.preprocessing.OneHotEncoder`
	- Ordinal Encoding: `sklearn.preprocessing.OrdinalEncoder`
	- LabelEncoder (similar to ordinal but only for target): `sklearn.preprocessing.Labelencoder`
	- Embeddings
7. Missing Values
	- SimpleImputer: `sklearn.impute.SimpleImputer`
	- KNNImputer: `sklearn.impute.KNNImputer`
8. Data Transformation / Mapping
	- KBinsDiscretizer: `sklearn.preprocessing.KBinsDiscretizer`
	- FunctionTransformer: `sklearn.preprocessing.FunctionTransformer`
	- Binarizer (Thresholding (turns numbers into 0/1)): `sklean.preprocessing.Binarizer`
9. Clustering
	- KMeans: use `sklearn.cluster.KMeans`, to create new feature, it would be a categorical feature. KMeans takes in `n_clusters`. Use dummies with this to get final features.
10. Text Transforms
	- CountVectorizer (bag of words): `sklearn.feature_extraction.text.CountVectorizer`
	- Tf-IDF Vectorizer: `sklearn.feature_extraction.text.TfidfVectorizer`
	- tfidf and bow: `sklearn.feature_extraction.text.TfidfTransformer`
	- Hashing Vectorizer: `sklearn.feature_extraction.text.HashingVectorizer`
11. Feature Selection
	- Keep the top k features: `sklearn.feature_selection.SelectKBest()` / `sklearn.feature_selection.SelectPercentile()`
	- Recursive feature elimination: `sklearn.feature_selection.RFE()` / `sklearn.feature_selection.RFECV()`
	- concatenate all the features obtained previously to get a new dataset. For concatenation use `pd.concat([], axis=1)`

## Other very cool stuff
1. `sklearn.compose.make_column_selector`: Automatically select numerical/categorical columns.
2. `sklearn.compose.make_column_selector`: Automatically select columns for column transformer.
3. `[any model].check_is_fitted()`: check if the model is trained.
4. `sklearn.model_selection.cross_val_score`: cross validation in one line.
## Model Selection

Use this flow chart
![[ml_map.svg]]
