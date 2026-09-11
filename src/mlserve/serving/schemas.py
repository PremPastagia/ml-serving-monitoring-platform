"""Pydantic request/response models, generated from the data contract.

The request model is *built* from ``mlserve.data.schema`` rather than hand-written.
That is deliberate: a hand-written API model is a second copy of the contract, and
the second copy is always the one that goes stale. Generating it means adding a
feature, tightening a range or renaming a level automatically changes what the API
accepts, and ``tests/test_api.py`` asserts the generated model still matches the
contract exactly.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, create_model

from mlserve.data.schema import (
    CATEGORICAL_FEATURES,
    FEATURES,
    MISSING_CATEGORY,
    NUMERIC_FEATURES,
    FeatureSpec,
)

#: Example values used for the OpenAPI example payload and for smoke tests.
EXAMPLE_RECORD: dict[str, Any] = {
    "age": 39,
    "workclass": "State-gov",
    "education_num": 13,
    "marital_status": "Never-married",
    "occupation": "Adm-clerical",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "capital_gain": 2174,
    "capital_loss": 0,
    "hours_per_week": 40,
    "native_country": "United-States",
}


def _field_for(spec: FeatureSpec) -> tuple[Any, Any]:
    """Map one contract feature onto an annotated Pydantic field."""
    if spec.is_numeric:
        constraints: dict[str, Any] = {"description": spec.description}
        if spec.minimum is not None:
            constraints["ge"] = spec.minimum
        if spec.maximum is not None:
            constraints["le"] = spec.maximum
        # int rather than float: every numeric column in this dataset is integral, and
        # accepting 39.7 as an age would silently change the binning at serving time.
        return (int, Field(..., **constraints))

    allowed = tuple(spec.allowed) + ((MISSING_CATEGORY,) if spec.nullable else ())
    annotation = Literal[allowed]  # type: ignore[valid-type]
    described = (
        f"{spec.description} Allowed values: {len(allowed)} level(s)."
        + (f" Use '{MISSING_CATEGORY}' when the value is unknown." if spec.nullable else "")
    )
    return (annotation, Field(..., description=described))


PredictionRecord: type[BaseModel] = create_model(  # type: ignore[call-overload]
    "PredictionRecord",
    __config__=ConfigDict(
        # An unexpected field is rejected, not ignored. Silently dropping an extra key
        # is how a client ships a renamed field into production and nobody notices.
        extra="forbid",
        json_schema_extra={"example": EXAMPLE_RECORD},
    ),
    **{spec.name: _field_for(spec) for spec in FEATURES},
)
PredictionRecord.__doc__ = (
    "One respondent's features. Field names, types, ranges and allowed categorical "
    "levels are generated from the training data contract."
)


class PredictRequest(BaseModel):
    """A batch of records to score. A single prediction is a batch of one."""

    model_config = ConfigDict(
        json_schema_extra={"example": {"records": [EXAMPLE_RECORD]}},
        extra="forbid",
    )

    records: Annotated[
        list[PredictionRecord],  # type: ignore[valid-type]
        Field(min_length=1, description="One or more records to score."),
    ]


class Prediction(BaseModel):
    """The scored result for one input record."""

    probability: float = Field(..., ge=0.0, le=1.0,
                               description="P(income > 50K) for this record.")
    prediction: int = Field(..., description="1 when probability >= threshold, else 0.")
    label: str = Field(..., description="The human-readable class label.")


class PredictResponse(BaseModel):
    request_id: str = Field(..., description="Correlates this response with the server logs.")
    model_name: str
    model_version: str = Field(..., description="Registry version actually used for these scores.")
    model_alias: str | None = None
    threshold: float
    n_records: int
    predictions: list[Prediction]
    latency_ms: float = Field(..., description="Server-side handling time for this request.")

    model_config = ConfigDict(protected_namespaces=())


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"] = Field(
        ..., description="'degraded' means the process is up but cannot serve predictions."
    )
    model_loaded: bool
    model_version: str | None = None
    uptime_seconds: float
    version: str = Field(..., description="mlserve package version.")

    model_config = ConfigDict(protected_namespaces=())


class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    model_alias: str | None
    model_source: str = Field(..., description="'registry' or 'file'.")
    run_id: str | None = None
    dataset_version: str | None = None
    code_version: str | None = None
    git_commit: str | None = None
    training_fingerprint: str | None = None
    loaded_at: str
    load_seconds: float
    input_columns: list[str] = Field(..., description="Exactly what POST /predict requires.")
    model_features: list[str] = Field(..., description="Features after internal engineering.")
    metrics: dict[str, float] = Field(default_factory=dict,
                                      description="Offline metrics recorded at training time.")

    model_config = ConfigDict(protected_namespaces=())


class ErrorDetail(BaseModel):
    type: str = Field(..., description="Stable machine-readable error class.")
    message: str
    details: list[dict[str, Any]] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    """Every non-2xx response from this service has this shape."""

    request_id: str
    error: ErrorDetail


def contract_field_names() -> list[str]:
    """The field names the generated request model exposes."""
    return list(PredictionRecord.model_fields.keys())


__all__ = [
    "CATEGORICAL_FEATURES",
    "EXAMPLE_RECORD",
    "NUMERIC_FEATURES",
    "ErrorDetail",
    "ErrorResponse",
    "HealthResponse",
    "ModelInfoResponse",
    "PredictRequest",
    "PredictResponse",
    "Prediction",
    "PredictionRecord",
    "contract_field_names",
]
