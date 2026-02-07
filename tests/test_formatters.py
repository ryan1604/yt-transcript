from yt_transcript.formatters import format_output
from yt_transcript.models import TranscriptResult, TranscriptSegment


def _sample_result() -> TranscriptResult:
    return TranscriptResult(
        video_id="dQw4w9WgXcQ",
        source="captions",
        language="en",
        segments=[
            TranscriptSegment(start=0.2, end=2.1, text="Hello world"),
            TranscriptSegment(start=62.2, end=64.9, text="Second line"),
        ],
    )


def test_txt_without_timestamps() -> None:
    output = format_output(_sample_result(), fmt="txt", timestamps=False)
    assert output == "Hello world\nSecond line\n"


def test_txt_with_timestamps() -> None:
    output = format_output(_sample_result(), fmt="txt", timestamps=True)
    assert output == "[00:00] Hello world\n[01:02] Second line\n"


def test_srt_format() -> None:
    output = format_output(_sample_result(), fmt="srt", timestamps=False)
    assert "1\n00:00:00,200 --> 00:00:02,100\nHello world" in output
    assert "2\n00:01:02,200 --> 00:01:04,900\nSecond line" in output


def test_json_with_timestamps() -> None:
    output = format_output(_sample_result(), fmt="json", timestamps=True)
    assert '"video_id": "dQw4w9WgXcQ"' in output
    assert '"start": 0.2' in output
    assert '"end": 2.1' in output
