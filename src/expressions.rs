use std::io::Cursor;
use wavers::{Samples, Wav, ReadSeek};
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

fn make_output_type(input_fields: &[Field]) -> PolarsResult<Field> {
    Ok(Field::new(
        input_fields[0].name().clone(),
        DataType::List(Box::new(DataType::Float32))
    ))
}

#[polars_expr(output_type_func=make_output_type)]
fn decode_wav(inputs: &[Series]) -> PolarsResult<Series> {
    let bins = inputs[0].binary()?;

    // Map each binary value to decoded samples
    let decoded: PolarsResult<Vec<Option<Vec<f32>>>> = bins
        .into_iter()
        .map(|bin_opt| {
            Ok(match bin_opt {
                Some(bin) => {
                    let cursor = Cursor::new(bin.to_vec());
                    let boxed: Box<dyn ReadSeek> = Box::new(cursor);
                    let mut reader = Wav::new(boxed).map_err(|e| {
                        PolarsError::ComputeError(format!("Construct WAV reader: {e}").into())
                    })?;
                    let samples : Samples<f32> = reader.read().map_err(|e| {
                        PolarsError::ComputeError(format!("Read WAV: {e}").into())
                    })?;
                    Some(samples.to_vec())
                }
                None => None,
            })
        })
        .collect();

    // Convert the decoded samples into a Series
    let decoded = decoded?;
    let total_values = decoded.iter().map(|arr_opt| {
        if let Some(arr) = arr_opt {
            arr.len()
        } else {
            0
        }
    }).sum();
    let mut builder = ListPrimitiveChunkedBuilder::<Float32Type>::new(
        inputs[0].name().clone(),
        decoded.len(),
        total_values,
        DataType::Float32,
    );

    for samples in decoded {
        match samples {
            Some(samples) => builder.append_iter(samples.into_iter().map(Some)),
            None => builder.append_null(),
        }
    }

    Ok(builder.finish().into_series())
}
