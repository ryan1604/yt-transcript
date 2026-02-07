import pytest

from yt_transcript.errors import InvalidInputError
from yt_transcript.url_parser import extract_video_id, normalize_video_url


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://youtu.be/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/shorts/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
    ],
)
def test_extract_video_id_valid(url: str, expected: str) -> None:
    assert extract_video_id(url) == expected


@pytest.mark.parametrize(
    "url",
    [
        "ftp://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://example.com/watch?v=dQw4w9WgXcQ",
        "https://www.youtube.com/watch?v=too_short",
        "not-a-url",
    ],
)
def test_extract_video_id_invalid(url: str) -> None:
    with pytest.raises(InvalidInputError):
        extract_video_id(url)


def test_normalize_video_url() -> None:
    assert normalize_video_url("dQw4w9WgXcQ") == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
