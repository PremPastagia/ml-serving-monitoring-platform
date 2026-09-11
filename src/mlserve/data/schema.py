"""The single source of truth for the data contract.

Everything downstream -- the validator, the feature pipeline, the FastAPI request
model, the drift detectors and the synthetic scenario generator -- reads this module.
Defining the contract once is what makes "the API rejects what training rejected"
true by construction rather than by convention.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

RAW_COLUMNS: list[str] = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income",
]

TARGET = "income"
POSITIVE_LABEL = ">50K"
NEGATIVE_LABEL = "<=50K"
TARGET_CLASSES = (NEGATIVE_LABEL, POSITIVE_LABEL)

#: Columns present in the raw file that are deliberately NOT used as features.
#:
#: ``fnlwgt`` is the Census Bureau's inverse-probability *sampling weight* for the
#: row: it describes how many people in the population the record stands for. It is
#: a property of the survey design, not of the person, so (a) it is unavailable at
#: serving time for a new applicant and (b) letting a tree split on it lets the model
#: learn the sampling frame instead of the income relationship. It is dropped.
#:
#: ``education`` is dropped because ``education_num`` is its exact ordinal encoding.
#: Keeping both is duplicated information that inflates one concept's weight in the
#: one-hot block and doubles that concept's contribution to aggregate drift scores.
DROPPED_COLUMNS: list[str] = ["fnlwgt", "education"]


class FeatureKind(str, Enum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"


@dataclass(frozen=True)
class FeatureSpec:
    """Declarative contract for one serving feature."""

    name: str
    kind: FeatureKind
    description: str
    # numeric constraints
    minimum: float | None = None
    maximum: float | None = None
    # categorical constraints
    allowed: tuple[str, ...] = ()
    #: values that mean "the respondent did not answer"; nullable features accept them
    nullable: bool = False

    @property
    def is_numeric(self) -> bool:
        return self.kind is FeatureKind.NUMERIC

    @property
    def is_categorical(self) -> bool:
        return self.kind is FeatureKind.CATEGORICAL


WORKCLASS = (
    "Private",
    "Self-emp-not-inc",
    "Self-emp-inc",
    "Federal-gov",
    "Local-gov",
    "State-gov",
    "Without-pay",
    "Never-worked",
)
MARITAL_STATUS = (
    "Married-civ-spouse",
    "Divorced",
    "Never-married",
    "Separated",
    "Widowed",
    "Married-spouse-absent",
    "Married-AF-spouse",
)
OCCUPATION = (
    "Tech-support",
    "Craft-repair",
    "Other-service",
    "Sales",
    "Exec-managerial",
    "Prof-specialty",
    "Handlers-cleaners",
    "Machine-op-inspct",
    "Adm-clerical",
    "Farming-fishing",
    "Transport-moving",
    "Priv-house-serv",
    "Protective-serv",
    "Armed-Forces",
)
RELATIONSHIP = ("Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried")
RACE = ("White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black")
SEX = ("Female", "Male")
NATIVE_COUNTRY = (
    "United-States", "Cambodia", "England", "Puerto-Rico", "Canada", "Germany",
    "Outlying-US(Guam-USVI-etc)", "India", "Japan", "Greece", "South", "China",
    "Cuba", "Iran", "Honduras", "Philippines", "Italy", "Poland", "Jamaica",
    "Vietnam", "Mexico", "Portugal", "Ireland", "France", "Dominican-Republic",
    "Laos", "Ecuador", "Taiwan", "Haiti", "Columbia", "Hungary", "Guatemala",
    "Nicaragua", "Scotland", "Thailand", "Yugoslavia", "El-Salvador",
    "Trinadad&Tobago", "Peru", "Hong", "Holand-Netherlands",
)

#: ``?`` is the raw file's missing marker for the three self-reported categoricals.
MISSING_MARKER = "?"
MISSING_CATEGORY = "__missing__"

FEATURES: list[FeatureSpec] = [
    FeatureSpec("age", FeatureKind.NUMERIC, "Age of the respondent in years.", minimum=17, maximum=90),
    FeatureSpec(
        "workclass", FeatureKind.CATEGORICAL,
        "Employment sector. Self-reported; '?' means not answered.",
        allowed=WORKCLASS, nullable=True,
    ),
    FeatureSpec(
        "education_num", FeatureKind.NUMERIC,
        "Highest education level as an ordinal rank (1=Preschool .. 16=Doctorate).",
        minimum=1, maximum=16,
    ),
    FeatureSpec("marital_status", FeatureKind.CATEGORICAL, "Marital status.", allowed=MARITAL_STATUS),
    FeatureSpec(
        "occupation", FeatureKind.CATEGORICAL,
        "Occupation category. Self-reported; '?' means not answered.",
        allowed=OCCUPATION, nullable=True,
    ),
    FeatureSpec("relationship", FeatureKind.CATEGORICAL, "Role within the household.", allowed=RELATIONSHIP),
    FeatureSpec("race", FeatureKind.CATEGORICAL, "Self-identified race.", allowed=RACE),
    FeatureSpec("sex", FeatureKind.CATEGORICAL, "Self-identified sex as recorded by the 1994 census.", allowed=SEX),
    FeatureSpec(
        "capital_gain", FeatureKind.NUMERIC,
        "Capital gains in USD for the year. Heavily zero-inflated and right-skewed.",
        minimum=0, maximum=99999,
    ),
    FeatureSpec(
        "capital_loss", FeatureKind.NUMERIC,
        "Capital losses in USD for the year. Heavily zero-inflated.",
        minimum=0, maximum=4356,
    ),
    FeatureSpec("hours_per_week", FeatureKind.NUMERIC, "Usual hours worked per week.", minimum=1, maximum=99),
    FeatureSpec(
        "native_country", FeatureKind.CATEGORICAL,
        "Country of origin. Self-reported; '?' means not answered.",
        allowed=NATIVE_COUNTRY, nullable=True,
    ),
]

FEATURE_NAMES: list[str] = [f.name for f in FEATURES]
NUMERIC_FEATURES: list[str] = [f.name for f in FEATURES if f.is_numeric]
CATEGORICAL_FEATURES: list[str] = [f.name for f in FEATURES if f.is_categorical]
BY_NAME: dict[str, FeatureSpec] = {f.name: f for f in FEATURES}


@dataclass(frozen=True)
class DatasetContract:
    """Bundles the contract so it can be serialised into run metadata."""

    features: list[FeatureSpec] = field(default_factory=lambda: list(FEATURES))
    target: str = TARGET
    positive_label: str = POSITIVE_LABEL

    def to_dict(self) -> dict:
        return {
            "target": self.target,
            "positive_label": self.positive_label,
            "dropped_columns": list(DROPPED_COLUMNS),
            "features": [
                {
                    "name": f.name,
                    "kind": f.kind.value,
                    "description": f.description,
                    "minimum": f.minimum,
                    "maximum": f.maximum,
                    "allowed": list(f.allowed),
                    "nullable": f.nullable,
                }
                for f in self.features
            ],
        }


CONTRACT = DatasetContract()
