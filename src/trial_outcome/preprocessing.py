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
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

from trial_outcome.config import RANDOM_SEED

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
# Optional columns from trial_outcome.data.top.TOP_FEATURE_DTYPES, encoded when present.
TOP_MULTI_VALUED_FEATURES = ["icd_chapters", "icd_categories"]
TEXT_FEATURES = ["criteria"]

MISSING = "missing"
SEPARATOR = "|"
DEFAULT_MIN_COUNT = 10
# Chosen by cross-validated log loss over train + valid: 50 beat 20 and matched 100.
DEFAULT_TEXT_COMPONENTS = 50


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


class TextComponents(TransformerMixin, BaseEstimator):
    """Dense components of one free-text column: TF-IDF, then truncated SVD, then scaling.

    The TF-IDF vocabulary holds words and two-word phrases (English stop words removed) found in
    at least ``min_df`` of the fitted rows and in at most 90% of them, capped at the
    ``max_features`` most frequent. Truncated SVD reduces it to ``n_components`` columns
    (latent semantic analysis), which are standardized. Missing text counts as empty. The
    output has at most one column fewer than the vocabulary has terms.
    """

    def __init__(
        self,
        n_components: int = DEFAULT_TEXT_COMPONENTS,
        min_df: int = DEFAULT_MIN_COUNT,
        max_features: int = 5000,
        random_state: int = RANDOM_SEED,
    ):
        self.n_components = n_components
        self.min_df = min_df
        self.max_features = max_features
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y=None) -> TextComponents:
        text = self._text(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = 1
        self.vectorizer_ = TfidfVectorizer(
            min_df=self.min_df,
            max_df=0.9,
            max_features=self.max_features,
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True,
        )
        try:
            tfidf = self.vectorizer_.fit_transform(text)
        except ValueError as error:  # no term passed min_df and max_df
            raise ValueError(f"the text vocabulary is too small: {error}") from error
        n_components = min(self.n_components, tfidf.shape[1] - 1)
        if n_components < 1:
            raise ValueError(f"the text vocabulary is too small: {tfidf.shape[1]} terms")
        self.svd_ = TruncatedSVD(n_components, random_state=self.random_state)
        self.scaler_ = StandardScaler().fit(self.svd_.fit_transform(tfidf))
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self)
        if not isinstance(X, pd.DataFrame) or list(X.columns) != list(self.feature_names_in_):
            raise ValueError(f"expected a DataFrame with columns {list(self.feature_names_in_)}")
        tfidf = self.vectorizer_.transform(self._text(X))
        return self.scaler_.transform(self.svd_.transform(tfidf))

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        check_is_fitted(self)
        column = self.feature_names_in_[0]
        return np.asarray(
            [f"{column}_text_{i}" for i in range(self.svd_.n_components)], dtype=object
        )

    @staticmethod
    def _text(X: pd.DataFrame) -> list[str]:
        if not isinstance(X, pd.DataFrame) or X.shape[1] != 1:
            raise ValueError("TextComponents expects a DataFrame with one text column")
        return X.iloc[:, 0].astype("string").fillna("").tolist()


class PresentColumns:
    """Column selector for :class:`~sklearn.compose.ColumnTransformer`: the listed columns that
    the fitted table has, so optional feature groups are encoded only when they're present.

    >>> PresentColumns(["criteria", "icd_chapters"])(pd.DataFrame(columns=["icd_chapters"]))
    ['icd_chapters']
    """

    def __init__(self, columns: list[str]):
        self.columns = list(columns)

    def __call__(self, X: pd.DataFrame) -> list[str]:
        return [column for column in self.columns if column in X.columns]


def make_preprocessor(
    min_count: int = DEFAULT_MIN_COUNT, text_components: int = DEFAULT_TEXT_COMPONENTS
) -> ColumnTransformer:
    """Encode the feature table as a dense float matrix with readable column names.

    - Categorical columns: one-hot, with missing values as their own ``missing`` category.
      Categories in fewer than ``min_count`` training rows, and categories first seen after
      fitting, share one ``infrequent`` column.
    - Multi-valued columns (``phases``, ``conditions``, ``intervention_types``, and TOP's
      ``icd_chapters`` and ``icd_categories``): one 0/1 column per value found in at least
      ``min_count`` training rows (:class:`MultiValueBinarizer`).
    - Numeric columns: training-median imputation, a missing-value indicator for each column
      with missing training values, then standardization.
    - TOP's ``criteria`` text: ``text_components`` dense columns (:class:`TextComponents`), from
      terms found in at least ``min_count`` training rows.

    The registry columns are required. TOP's columns are optional: each is encoded only if the
    table being fitted has it, so datasets built with ``--registry-only`` still work. Once
    fitted, the preprocessor expects the same columns.

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
            (
                "top_multi_valued",
                MultiValueBinarizer(min_count=min_count),
                PresentColumns(TOP_MULTI_VALUED_FEATURES),
            ),
            (
                "text",
                TextComponents(n_components=text_components, min_df=min_count),
                PresentColumns(TEXT_FEATURES),
            ),
        ],
        sparse_threshold=0.0,
        verbose_feature_names_out=False,
    )


def _categories_as_text(X: pd.DataFrame) -> pd.DataFrame:
    return X.astype("string").fillna(MISSING).astype(object)


def _as_float(X: pd.DataFrame) -> np.ndarray:
    return X.astype("Float64").to_numpy(dtype="float64", na_value=np.nan)
