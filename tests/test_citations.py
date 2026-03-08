"""Tests for citation manipulation helpers."""
from citations import offset_citations, validate_citations


class TestOffsetCitations:
    def test_single(self):
        assert offset_citations("see [1] and [3]", 100) == "see [101] and [103]"

    def test_comma(self):
        assert offset_citations("see [1, 3, 5]", 100) == "see [101, 103, 105]"

    def test_adjacent(self):
        assert offset_citations("[1][2][3]", 50) == "[51][52][53]"

    def test_zero_offset(self):
        assert offset_citations("see [1]", 0) == "see [1]"

    def test_no_citations(self):
        assert offset_citations("no citations here", 100) == "no citations here"

    def test_large_numbers_no_collision(self):
        # [12] must not be partially matched by [1] replacement
        assert offset_citations("[12] then [1]", 400) == "[412] then [401]"

    def test_mixed_comma_and_single(self):
        result = offset_citations("see [1, 2] and [3]", 10)
        assert result == "see [11, 12] and [13]"


class TestValidateCitations:
    def test_strips_out_of_range(self):
        result = validate_citations("see [1] and [999]", 100)
        assert result == "see [1] and"

    def test_comma_partial(self):
        assert validate_citations("see [1, 999]", 100) == "see [1]"

    def test_all_valid(self):
        assert validate_citations("see [1] and [50]", 100) == "see [1] and [50]"

    def test_whitespace_cleanup(self):
        assert validate_citations("data [999] here", 100) == "data here"

    def test_trailing_space_stripped(self):
        assert validate_citations("end [999]", 100) == "end"

    def test_comma_all_invalid(self):
        assert validate_citations("see [998, 999]", 100) == "see"

    def test_zero_boundary(self):
        # [0] is out of range (valid is 1..max)
        assert validate_citations("[0] text [1]", 5) == "text [1]"
