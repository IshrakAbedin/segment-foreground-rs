use anyhow::{Context, Result, anyhow};
use clap::{Parser, ValueEnum};
use segment_foreground_rs::{run_modnet, run_u2net};
use std::{env, path::PathBuf};

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Model to use for segmentation
    #[arg(value_enum, long, short, default_value_t = Model::Modnet)]
    model: Model,

    /// Path to the input file
    #[arg(short, long, value_parser = validate_file_exists)]
    input: PathBuf,

    /// Path to the output file
    #[arg(short, long, default_value = "matte.png")]
    output: PathBuf,

    /// Number of intra-op threads for ORT
    #[arg(long, default_value_t = 4)]
    pub threads: usize,

    /// Try to register CUDA EP (requires building with --features cuda)
    #[arg(long, default_value_t = false)]
    pub use_cuda: bool,

    /// Try to register TensorRT EP (requires --features tensorrt). If both TRT and CUDA enabled,
    /// TensorRT is preferred first, then CUDA as fallback.
    #[arg(long, default_value_t = false)]
    pub use_tensorrt: bool,

    /// Try to register DirectML EP (requires building with --features directml)
    #[arg(long, default_value_t = false)]
    pub use_directml: bool,

    /// Device (GPU) ID to be used with CUDA, TensorRT, or DirectML
    #[arg(short, long, default_value_t = 0)]
    pub device_id: i32,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
enum Model {
    /// Use MODNet for human subject detection
    Modnet,
    /// Use U-Square-Net for general salient object detection
    U2net,
}

fn main() -> Result<()> {
    let args = Cli::parse();

    match &args.model {
        Model::Modnet => {
            let model_path = get_model_path_with_fallback("models/modnet.onnx")?;
            run_modnet(
                model_path.to_str().unwrap(),
                args.input.to_str().unwrap(),
                args.output.to_str().unwrap(),
                args.threads,
                args.use_cuda,
                args.use_tensorrt,
                args.use_directml,
                args.device_id,
            )?;
        }
        Model::U2net => {
            let model_path = get_model_path_with_fallback("models/u2net.onnx")?;
            run_u2net(
                model_path.to_str().unwrap(),
                args.input.to_str().unwrap(),
                args.output.to_str().unwrap(),
                args.threads,
                args.use_cuda,
                args.use_tensorrt,
                args.use_directml,
                args.device_id,
            )?;
        }
    }

    Ok(())
}

fn validate_file_exists(path: &str) -> Result<PathBuf, String> {
    let path = PathBuf::from(path);
    if path.exists() && path.is_file() {
        Ok(path)
    } else {
        Err(format!("File does not exist: {}", path.display()))
    }
}

fn get_model_path(model_name: impl AsRef<str>) -> Result<PathBuf> {
    let exe_path = env::current_exe()?;
    let exe_dir = exe_path
        .parent()
        .ok_or(anyhow!("Could not get parent directory of executable"))?;

    Ok(exe_dir.join(model_name.as_ref()))
}

fn get_model_path_with_fallback(model_name: impl AsRef<str>) -> Result<PathBuf> {
    let model_name = model_name.as_ref();

    get_model_path(model_name)
        .and_then(|path| validate_file_exists(path.to_str().unwrap()).map_err(|e| anyhow!(e)))
        .or_else(|_| validate_file_exists(model_name).map_err(|e| anyhow!(e)))
        .with_context(|| format!("Cannot find the model {}", model_name))
}
