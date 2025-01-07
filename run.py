import polars as pl
from wav_decoder import decode_wav


df = pl.DataFrame(
    {
        "wav": [open("tests/test.wav", "rb").read()],
    }
)
result = df.with_columns(decode_wav("wav"))
print(result)
