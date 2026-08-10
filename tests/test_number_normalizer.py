"""NumberNormalizer の単体テストモジュール。"""

from lumi_companion.audio.number_normalizer import NumberNormalizer


def test_normalize_numbers_static() -> None:
    # Arrange
    raw = "第I章 十個のりんご ①番 １２３"

    # Act
    res = NumberNormalizer.normalize(raw)

    # Assert
    assert res == "第１章 １０個のりんご １番 １２３"
