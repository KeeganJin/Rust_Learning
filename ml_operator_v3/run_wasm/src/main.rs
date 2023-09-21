#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
use wasmedge_sdk::{
    config::{CommonConfigOptions, ConfigBuilder, HostRegistrationConfigOptions},
    params,
    dock::{Param,VmDock},
    plugin::PluginManager,
    Module, VmBuilder,
};

use std::fs::File;
use std::io::Read;
use std::env;

struct InferenceResult(usize, f32);

fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    infer()?;
    println!("first inference");

    // println!("second inference");
    // infer()?;

    Ok(())
}


#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
fn infer() -> Result<(), Box<dyn std::error::Error>> {

    //works in the Terminal
    let dir_mapping = &String::from(".:.");
    // let wasm_file = &String::from("wasmedge-wasinn-example-mobilenet-image.wasm");
    let wasm_file = &String::from("target/wasm32-wasi/debug/ml_pytorch_lib.wasm");
    // let wasm_file = &String::from("ml_pytorch.wasm");

    let model_bin = &String::from("mobilenet.pt");
    let image_file = &String::from("input.jpg");
    // let args: Vec<String> = vec![dir_mapping,wasm_file,model_bin,image_file];

    //image to tensor
    let image_tensor_data = image_to_tensor(image_file.to_string(), 224, 224);

    println!("load plugin");

    // load wasinn-pytorch-plugin from the default plugin directory: /usr/local/lib/wasmedge
    PluginManager::load(None)?;

    let config = ConfigBuilder::new(CommonConfigOptions::default())
        .with_host_registration_config(HostRegistrationConfigOptions::default().wasi(true))
        .build()?;
    assert!(config.wasi_enabled());
    // assert!(config.wasi_nn_enabled());

    // load wasm module from file
    let module = Module::from_file(Some(&config), wasm_file)?;

//    // build a Vm
//     let mut vm = VmBuilder::new()
//         .with_config(config)
//         .with_plugin_wasi_nn()
//         .build()?
//         .register_module(Some("extern"), module)?;

    // build a Vm
    let mut vm = VmBuilder::new()
    .with_config(config)
    .with_plugin_wasi_nn()
    .build()?
    .register_module(None, module)?;

    // init wasi module
    vm.wasi_module_mut()
        // .ok_or("Not found wasi module")?
        .expect("Not found wasi module")
        .initialize(
            // remove image_file, load it from wasm module
            // Some(vec![wasm_file, model_bin, image_file]),
            Some(vec![wasm_file, model_bin]),
            None,
            Some(vec![dir_mapping]),
        );


    // vm.run_func(Some("extern"), "_start", params!())?;

    let vm = VmDock::new(vm);

    // call the call_infer in ml_pytorch_lib.wasm
    // vm.run_func("call_infer", params!())?;


    let params = image_tensor_data;
    // let params = "This is an important message".as_bytes().to_vec();
    // check the data type of params
    let params = vec![Param::VecU8(&params)];


    match vm.run_func("call_infer", params)? {
        Ok(mut res) => {
            println!(
                "Run bindgen -- call_infer: {:?}",
                // res.pop().unwrap().downcast::<Vec<u8>>().unwrap()
                res.pop().unwrap().downcast::<f32>().unwrap()
            );
        }
        Err(err) => {
            println!("Run bindgen -- call_infer FAILED {}", err);
        }
    }


    // vm.run_func(Some("extern"), "_start", params!())?;

    Ok(())
}

fn image_to_tensor(path: String, height: u32, width: u32) -> Vec<u8> {
    let mut file_img = File::open(path).unwrap();
    let mut img_buf = Vec::new();
    file_img.read_to_end(&mut img_buf).unwrap();
    let img = image::load_from_memory(&img_buf).unwrap().to_rgb8();
    let resized =
        image::imageops::resize(&img, height, width, ::image::imageops::FilterType::Triangle);
    let mut flat_img: Vec<f32> = Vec::new();
    for rgb in resized.pixels() {
        flat_img.push((rgb[0] as f32 / 255. - 0.485) / 0.229);
        flat_img.push((rgb[1] as f32 / 255. - 0.456) / 0.224);
        flat_img.push((rgb[2] as f32 / 255. - 0.406) / 0.225);
    }
    let bytes_required = flat_img.len() * 4;
    let mut u8_f32_arr: Vec<u8> = vec![0; bytes_required];

    for c in 0..3 {
        for i in 0..(flat_img.len() / 3) {
            // Read the number as a f32 and break it into u8 bytes
            let u8_f32: f32 = flat_img[i * 3 + c] as f32;
            let u8_bytes = u8_f32.to_ne_bytes();

            for j in 0..4 {
                u8_f32_arr[((flat_img.len() / 3 * c + i) * 4) + j] = u8_bytes[j];
            }
        }
    }
    return u8_f32_arr;
}