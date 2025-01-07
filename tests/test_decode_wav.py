import polars as pl
import soundfile as sf
from wav_decoder import decode_wav



def test_decode_wav():
    with open("tests/test.wav", "rb") as f:
        wav_data = f.read()
        df = pl.DataFrame({"wav": [wav_data]})
    
    data = sf.read("tests/test.wav")
    result = df.with_columns(decode_wav("wav"))
    expected_df = pl.DataFrame(
        {
            "wav": [data]
        }
    )
    assert result.equals(expected_df)
