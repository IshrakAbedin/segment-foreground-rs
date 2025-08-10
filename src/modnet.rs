use anyhow::{Result, anyhow};
use image::{GrayImage, ImageReader, Luma, imageops::FilterType};
use ndarray::{Ix3, Ix4};
use ort::session::{Session, builder::GraphOptimizationLevel};
use ort::value::Tensor;

use crate::preprocess::{preprocess_modnet_nchw, resize_with_padding};

/// Run MODNet (positional input) — model_path, input_path, output_path
pub fn run_modnet(model_path: &str, input_path: &str, output_path: &str) -> Result<()> {
    let mut session = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .commit_from_file(model_path)?;

    // println!(
    //     "Model inputs: {:?}",
    //     session.inputs.iter().map(|i| &i.name).collect::<Vec<_>>()
    // );
    // println!(
    //     "Model outputs: {:?}",
    //     session.outputs.iter().map(|o| &o.name).collect::<Vec<_>>()
    // );

    let img = ImageReader::open(input_path)?.decode()?.to_rgb8();
    let (padded_img, (pad_x, pad_y, resized_w, resized_h)) = resize_with_padding(&img, 512, 512);
    let input_arr = preprocess_modnet_nchw(&padded_img);
    let input_tensor = Tensor::from_array(input_arr)?;

    // positional input (works regardless of input name)
    let outputs = session.run(ort::inputs![input_tensor])?;

    // extract first output as f32 array view
    let arr_view = outputs[0].try_extract_array::<f32>()?;
    let arr_owned = arr_view.to_owned();

    // Support either 4D (1,1,H,W) or 3D (1,H,W)
    let alpha4 = match arr_owned.ndim() {
        4 => arr_owned.into_dimensionality::<Ix4>()?,
        3 => {
            let a3 = arr_owned.into_dimensionality::<Ix3>()?;
            // a3 shape: (1, H, W) -> create (1,1,H,W)
            let (b, h, w) = (a3.shape()[0], a3.shape()[1], a3.shape()[2]);
            let mut out = ndarray::Array4::<f32>::zeros((b, 1, h, w));
            for bi in 0..b {
                for y in 0..h {
                    for x in 0..w {
                        out[[bi, 0, y, x]] = a3[[bi, y, x]];
                    }
                }
            }
            out
        }
        d => {
            return Err(anyhow!(
                "Unexpected output dimensionality from model: {}",
                d
            ));
        }
    };

    // Build a 512x512 grayscale matte image
    let mut matte_full = GrayImage::new(512, 512);
    for y in 0..512 {
        for x in 0..512 {
            let val = alpha4[[0, 0, y as usize, x as usize]].clamp(0.0, 1.0);
            let byte = (val * 255.0).round() as u8;
            matte_full.put_pixel(x, y, Luma([byte]));
        }
    }

    // Crop out padding
    let matte_cropped = image::imageops::crop_imm(
        &matte_full,
        pad_x.into(),
        pad_y.into(),
        resized_w,
        resized_h,
    )
    .to_image();

    // Resize back to original input size
    let matte_final = image::imageops::resize(
        &matte_cropped,
        img.width(),
        img.height(),
        FilterType::Lanczos3,
    );

    matte_final.save(output_path)?;
    println!("Saved MODNet alpha to {}", output_path);

    Ok(())
}
