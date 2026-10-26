import pandas as pd
import pytest

from proyecto_ml.anonymize import anonymize, hash_value

SALT = "sal-de-prueba-1234567890"


def test_hash_determinista_y_dependiente_de_sal():
    assert hash_value("1037000000", SALT) == hash_value(" 1037000000 ", SALT)
    assert hash_value("1037000000", SALT) != hash_value("1037000000", SALT + "x")


def test_sal_corta_falla():
    with pytest.raises(ValueError):
        hash_value("1", "corta")


def test_anonymize_elimina_pii_y_agrupa():
    df = pd.DataFrame({"cedula": ["1", "2"], "nombre": ["A", "B"], "edad": [23, 41]})
    out = anonymize(df, id_columns=["cedula"], drop_columns=["nombre"], bins={"edad": [18, 30, 50]}, salt=SALT)
    assert "nombre" not in out.columns
    assert out["cedula"].str.len().eq(16).all()
    assert out["edad"].tolist() == ["(17.999, 30.0]", "(30.0, 50.0]"]
