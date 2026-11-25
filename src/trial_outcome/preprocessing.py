"""Model preprocessing: the typed feature table -> a numeric matrix.

:func:`make_preprocessor` is the first step of every model pipeline in
:mod:`trial_outcome.modeling`, so it is fitted on the training rows only. Validation and test
trials never influence the category vocabularies, the imputed values, or the scaling.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

# Together these cover every column in trial_outcome.features.FEATURE_DTYPES.
CATEGORICAL_FEATURES = [
    "study_type",
    "allocation",
    "intervention_model",
    "primary_purpose",
    "masking",
    "sponsor_class",
    "sex",
    "healthy_volunteers",
]
MULTI_VALUED_FEATURES = ["phases", "conditions", "intervention_types"]
NUMERIC_FEATURES = ["minimum_age_years", "maximum_age_years", "arm_count"]

MISSING = "missing"
SEPARATOR = "|"
DEFAULT_MIN_COUNT = 10


def normalize_token(token: str) -> str:
    """Lowercase a category value and collapse its whitespace.

    >>> normalize_token("  Breast   Cancer ")
    'breast cancer'
    """
    return " ".join(token.lower().split())


class MultiValueBinarizer(TransformerMixin, BaseEstimator):
    """One 0/1 column per value of ``|``-joined multi-valued columns.

    Values are normalized with :func:`normalize_token`. A value gets a column only if it appears
    in at least ``min_count`` of the rows seen by :meth:`fit`; rarer values and values first seen
    by :meth:`transform` are dropped. A missing cell has no values, so all its columns are 0.

    >>> frame = pd.DataFrame({"conditions": ["Asthma|COPD", "asthma", pd.NA]})
    >>> binarizer = MultiValueBinarizer(min_count=2).fit(frame)
    >>> binarizer.get_feature_names_out().tolist()
    ['conditions=asthma']
    >>> binarizer.transform(frame).tolist()
    [[1.0], [1.0], [0.0]]
    """

    def __init__(self, min_count: int = 1, separator: str = SEPARATOR):
        self.min_count = min_count
        self.separator = separator

    def fit(self, X: pd.DataFrame, y=None) -> MultiValueBinarizer:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("MultiValueBinarizer expects a DataFrame")
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        self.vocabulary_ = {}
        for column in X.columns:
            counts = Counter(value for values in self._split(X[column]) for value in values)
            self.vocabulary_[column] = sorted(v for v, n in counts.items() if n >= self.min_count)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self)
        if not isinstance(X, pd.DataFrame) or list(X.columns) != list(self.feature_names_in_):
            raise ValueError(f"expected a DataFrame with columns {list(self.feature_names_in_)}")
        blocks = []
        for column in self.feature_names_in_:
            position = {value: j for j, value in enumerate(self.vocabulary_[column])}
            block = np.zeros((len(X), len(position)))
            for i, values in enumerate(self._split(X[column])):
                for value in values:
                    if value in position:
                        block[i, position[value]] = 1.0
            blocks.append(block)
        return np.hstack(blocks)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        check_is_fitted(self)
        names = [f"{c}={v}" for c in self.feature_names_in_ for v in self.vocabulary_[c]]
        return np.asarray(names, dtype=object)

    def _split(self, cells: pd.Series) -> list[set[str]]:
        return [
            set()
            if pd.isna(cell)
            else {normalize_token(v) for v in str(cell).split(self.separator)} - {""}
            for cell in cells
        ]


def make_preprocessor(min_count: int = DEFAULT_MIN_COUNT) -> ColumnTransformer:
    """Encode the feature table as a dense float matrix with readable column names.

    - Categorical columns: one-hot, with missing values as their own ``missing`` category.
      Categories in fewer than ``min_count`` training rows, and categories first seen after
      fitting, share one ``infrequent`` column.
    - Multi-valued columns (``phases``, ``conditions``, ``intervention_types``): one 0/1 column
      per value found in at least ``min_count`` training rows (:class:`MultiValueBinarizer`).
    - Numeric columns: training-median imputation, a missing-value indicator for each column
      with missing training values, then standardization.

    The output is dense because XGBoost reads the absent entries of a sparse matrix as missing,
    not as zero.
    """
    categorical = make_pipeline(
        FunctionTransformer(_categories_as_text, feature_names_out="one-to-one"),
        OneHotEncoder(
            handle_unknown="infrequent_if_exist", min_frequency=min_count, sparse_output=False
        ),
    )
    numeric = make_pipeline(
        FunctionTransformer(_as_float, feature_names_out="one-to-one"),
        SimpleImputer(strategy="median", add_indicator=True),
        StandardScaler(),
    )
    return ColumnTransformer(
        [
            ("categorical", categorical, CATEGORICAL_FEATURES),
            ("multi_valued", MultiValueBinarizer(min_count=min_count), MULTI_VALUED_FEATURES),
            ("numeric", numeric, NUMERIC_FEATURES),
        ],
        sparse_threshold=0.0,
        verbose_feature_names_out=False,
    )


def _categories_as_text(X: pd.DataFrame) -> pd.DataFrame:
    return X.astype("string").fillna(MISSING).astype(object)


def _as_float(X: pd.DataFrame) -> np.ndarray:
    return X.astype("Float64").to_numpy(dtype="float64", na_value=np.nan)
