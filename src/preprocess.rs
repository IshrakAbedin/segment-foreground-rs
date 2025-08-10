use image::{RgbImage, imageops::FilterType};
use ndarray::Array4;

/// Resize while preserving aspect ratio and pad (black) to target size.
/// Returns (padded_image, (pad_x, pad_y, resized_w, resized_h))
pub fn resize_with_padding(
    img: &RgbImage,
    target_w: u32,
    target_h: u32,
) -> (RgbImage, (u32, u32, u32, u32)) {
    let (orig_w, orig_h) = img.dimensions();
    let scale = f32::min(
        target_w as f32 / orig_w as f32,
        target_h as f32 / orig_h as f32,
    );
    let new_w = (orig_w as f32 * scale).round() as u32;
    let new_h = (orig_h as f32 * scale).round() as u32;
    let resized = image::imageops::resize(img, new_w, new_h, FilterType::Lanczos3);

    let pad_x = (target_w - new_w) / 2;
    let pad_y = (target_h - new_h) / 2;

    let mut padded = RgbImage::new(target_w, target_h);
    // fill with black
    for (_x, _y, pixel) in padded.enumerate_pixels_mut() {
        *pixel = image::Rgb([0, 0, 0]);
    }
    image::imageops::overlay(&mut padded, &resized, pad_x.into(), pad_y.into());
    (padded, (pad_x, pad_y, new_w, new_h))
}

/// MODNet preprocessing: resize/pad must be done before calling this.
/// Converts an RGB image into NCHW Array4<f32> normalized to [-1, 1].
pub fn preprocess_modnet_nchw(img: &RgbImage) -> Array4<f32> {
    let (w, h) = (img.width() as usize, img.height() as usize);
    let mut data = Vec::with_capacity(1 * 3 * h * w);
    for c in 0..3 {
        for y in 0..h {
            for x in 0..w {
                let px = img.get_pixel(x as u32, y as u32);
                let v = px[c] as f32;
                data.push((v - 127.5) / 127.5_f32);
            }
        }
    }
    Array4::from_shape_vec((1, 3, h, w), data).expect("shape must match")
}

/// U²-Net preprocessing: ImageNet mean/std normalization, expects inputs scaled [0,1].
/// Input should already be resized/padded to target (320).
pub fn preprocess_u2net_nchw(img: &RgbImage) -> Array4<f32> {
    let mean = [0.485_f32, 0.456_f32, 0.406_f32];
    let std = [0.229_f32, 0.224_f32, 0.225_f32];

    let (w, h) = (img.width() as usize, img.height() as usize);
    let mut data = Vec::with_capacity(1 * 3 * h * w);
    for c in 0..3 {
        for y in 0..h {
            for x in 0..w {
                let px = img.get_pixel(x as u32, y as u32);
                let v = (px[c] as f32) / 255.0_f32;
                data.push((v - mean[c]) / std[c]);
            }
        }
    }
    Array4::from_shape_vec((1, 3, h, w), data).expect("shape must match")
}
