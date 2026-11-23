"""La validación fuera de tiempo no se ofrece con un único corte."""

from pathlib import Path


def test_no_hay_comando_fuera_de_tiempo():
    makefile = Path("Makefile").read_text(encoding="utf-8")
    assert "fuera-de-tiempo:" not in makefile
