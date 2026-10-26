import pandas as pd
import pytest

from proyecto_ml.synthetic import make_synthetic


@pytest.fixture(scope="session")
def synthetic_df() -> pd.DataFrame:
    return make_synthetic(n_rows=800, random_state=0, base_rate=0.1)
