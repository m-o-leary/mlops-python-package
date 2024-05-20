"""Define an interface for trainable machine learning models."""

# %% IMPORTS

import abc
import typing as T

import pydantic as pdt

from mlopskit import schema

# Model params
ParamKey = str
ParamValue = T.Any
Params = dict[ParamKey, ParamValue]


# %% MODELS


class Model(abc.ABC, pdt.BaseModel, strict=True, frozen=False, extra="forbid"):
    """Base class for a project model.

    Use a model to adapt AI/ML frameworks.
    e.g., to swap easily one model with another.
    """

    KIND: str

    def get_params(self, deep: bool = True) -> Params:
        """Get the model params.

        Args:
            deep (bool, optional): ignored.

        Returns:
            Params: internal model parameters.
        """
        params: Params = {}
        for key, value in self.model_dump().items():
            if not key.startswith("_") and not key.isupper():
                params[key] = value
        return params

    def set_params(self, **params: ParamValue) -> T.Self:
        """Set the model params in place.

        Returns:
            T.Self: instance of the model.
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self

    @abc.abstractmethod
    def fit(
        self,
        inputs: schema.BaseSchemaType[schema.TSchema],
        targets: schema.BaseSchemaType[schema.TSchema],
    ) -> T.Self:
        """Fit the model on the given inputs and targets.

        Args:
            inputs (schema.BaseSchemaType[schema.TSchema]): model training inputs.
            targets (schema.BaseSchemaType[schema.TSchema]): model training targets.

        Returns:
            T.Self: instance of the model.
        """

    @abc.abstractmethod
    def predict(
        self, inputs: schema.BaseSchemaType[schema.TSchema]
    ) -> schema.BaseSchemaType[schema.TSchema]:
        """Generate outputs with the model for the given inputs.

        Args:
            inputs (schema.BaseSchemaType[schema.TSchema]): model prediction inputs.

        Returns:
            schema.BaseSchemaType: model prediction outputs.
        """

    def get_internal_model(self) -> T.Any:
        """Return the internal model in the object.

        Raises:
            NotImplementedError: method not implemented.

        Returns:
            T.Any: any internal model (either empty or fitted).
        """
        raise NotImplementedError()
