import pandas as pd
from src.pipelines.data_quality_checks import validate_not_empty, validate_required_columns


def test_required_columns_pass():
    df = pd.DataFrame({"a": [1], "b": [2]})
    validate_not_empty(df, "demo")
    validate_required_columns(df, ["a", "b"], "demo")
