"""Define trainable machine learning models."""

# %% IMPORTS

import typing as T

from sklearn import compose, ensemble, pipeline, preprocessing

from bikes.core import schemas
from mlopskit import model

# %% MODELS


class BaselineSklearnModel(model.Model):
    """Simple baseline model based on scikit-learn.

    Parameters:
        max_depth (int): maximum depth of the random forest.
        n_estimators (int): number of estimators in the random forest.
        random_state (int, optional): random state of the machine learning pipeline.
    """

    KIND: T.Literal["BaselineSklearnModel"] = "BaselineSklearnModel"

    # params
    max_depth: int = 20
    n_estimators: int = 200
    random_state: int | None = 42
    # private
    _pipeline: pipeline.Pipeline | None = None
    _numericals: list[str] = [
        "yr",
        "mnth",
        "hr",
        "holiday",
        "weekday",
        "workingday",
        "temp",
        "atemp",
        "hum",
        "windspeed",
        "casual",
        # "registered", # too correlated with target
    ]
    _categoricals: list[str] = [
        "season",
        "weathersit",
    ]

    def fit(self, inputs: schemas.Inputs, targets: schemas.Targets) -> "BaselineSklearnModel":
        # subcomponents
        categoricals_transformer = preprocessing.OneHotEncoder(
            sparse_output=False, handle_unknown="ignore"
        )
        # components
        transformer = compose.ColumnTransformer(
            [
                ("categoricals", categoricals_transformer, self._categoricals),
                ("numericals", "passthrough", self._numericals),
            ],
            remainder="drop",
        )
        regressor = ensemble.RandomForestRegressor(
            max_depth=self.max_depth, n_estimators=self.n_estimators, random_state=self.random_state
        )
        # pipeline
        self._pipeline = pipeline.Pipeline(
            steps=[
                ("transformer", transformer),
                ("regressor", regressor),
            ]
        )
        self._pipeline.fit(X=inputs, y=targets[schemas.TargetsSchema.cnt])
        return self

    def predict(self, inputs: schemas.Inputs) -> schemas.Outputs:
        model = self.get_internal_model()
        prediction = model.predict(inputs)
        outputs = schemas.Outputs(
            {schemas.OutputsSchema.prediction: prediction}, index=inputs.index
        )
        return outputs

    def get_internal_model(self) -> pipeline.Pipeline:
        model = self._pipeline
        if model is None:
            raise ValueError("Model is not fitted yet!")
        return model


ModelKind = BaselineSklearnModel
