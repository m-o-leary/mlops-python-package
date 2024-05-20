"""Define and validate dataframe schemas."""

# %% IMPORTS

import pandera as pa
import pandera.typing as papd
import pandera.typing.common as padt

from mlopskit import schema

# %% SCHEMAS


class InputsSchema(schema.Schema):
    """Schema for the project inputs."""

    instant: papd.Index[padt.UInt32] = pa.Field(ge=0, check_name=True)
    dteday: papd.Series[padt.DateTime] = pa.Field()
    season: papd.Series[padt.UInt8] = pa.Field(isin=[1, 2, 3, 4])
    yr: papd.Series[padt.UInt8] = pa.Field(ge=0, le=1)
    mnth: papd.Series[padt.UInt8] = pa.Field(ge=1, le=12)
    hr: papd.Series[padt.UInt8] = pa.Field(ge=0, le=23)
    holiday: papd.Series[padt.Bool] = pa.Field()
    weekday: papd.Series[padt.UInt8] = pa.Field(ge=0, le=6)
    workingday: papd.Series[padt.Bool] = pa.Field()
    weathersit: papd.Series[padt.UInt8] = pa.Field(ge=1, le=4)
    temp: papd.Series[padt.Float16] = pa.Field(ge=0, le=1)
    atemp: papd.Series[padt.Float16] = pa.Field(ge=0, le=1)
    hum: papd.Series[padt.Float16] = pa.Field(ge=0, le=1)
    windspeed: papd.Series[padt.Float16] = pa.Field(ge=0, le=1)
    casual: papd.Series[padt.UInt32] = pa.Field(ge=0)
    registered: papd.Series[padt.UInt32] = pa.Field(ge=0)


Inputs = papd.DataFrame[InputsSchema]


class TargetsSchema(schema.Schema):
    """Schema for the project target."""

    instant: papd.Index[padt.UInt32] = pa.Field(ge=0, check_name=True)
    cnt: papd.Series[padt.UInt32] = pa.Field(ge=0)


Targets = papd.DataFrame[TargetsSchema]


class OutputsSchema(schema.Schema):
    """Schema for the project output."""

    instant: papd.Index[padt.UInt32] = pa.Field(ge=0, check_name=True)
    prediction: papd.Series[padt.UInt32] = pa.Field(ge=0)


Outputs = papd.DataFrame[OutputsSchema]
