# Segment-Foreground-Rust

A command-line tool to segment image foreground using MODNet or U-Square-Net

## Building and running the project

### Building

You can simply build the project using `cargo build` for debug mode and `cargo build -r` for release mode. However, you need to supply the application with the `ONNX` models of MODNet and U-Square-Net by putting them under the root directory of your project inside the `./models/` folder, or in the residing directory of your executable under the `./models/` folder as `modnet.onnx` and `u2net.onnx`.

I fetched the `ONNX` files from the following links:

1. MODNet from [the Google Drive link shared in the official repository](https://drive.google.com/file/d/1cgycTQlYXpTh26gB9FTnthE7AvruV8hd/view?usp=sharing)
2. U2Net from [the rmbg 🤗 Hugging Face page](https://huggingface.co/tomjackson2023/rembg/blob/main/u2net.onnx)

### Running
Once you have the project compiled and the models placed properly, you can run it either using

```sh
cargo run -- [OPTIONS]
```

or, if you have taken the binary out then

```sh
# Windows will have .exe after the app name
segment-foreground-rs [OPTIONS]
```

The options are:

```
Options:
  -m, --model <MODEL>
          Model to use for segmentation

          Possible values:
          - modnet: Use MODNet for human subject detection
          - u2net:  Use U-Square-Net for general salient object detection

          [default: modnet]

  -i, --input <INPUT>
          Path to the input file

  -o, --output <OUTPUT>
          Path to the output file

          [default: matte.png]

  -h, --help
          Print help (see a summary with '-h')

  -V, --version
          Print version
```

Only the `--input` argument is positional/mandatory. 

## Note
> I have let `ort` crate handle the ONNX Runtime. If it creates problem, you might want to look into how to link your own dynamic libraries for ONNX.
