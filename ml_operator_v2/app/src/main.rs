use std::{fs::File, io::Read, path::Path};
use image;
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
use wasmedge_sdk::{
    config::{CommonConfigOptions, ConfigBuilder, HostRegistrationConfigOptions},
    dock::{Param,VmDock},
    plugin::PluginManager,
    Module, VmBuilder,
};




fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    infer()?;
    Ok(())
}


#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
fn infer() -> Result<(), Box<dyn std::error::Error>> {


    let wasm_file = &String::from("rust_mobilenet_food_lib.wasm");
    let image_name = &String::from("banana.jpg");

    //image to tensor
    let image_tensor_data = image_to_tensor(image_name.to_string(), 192, 192);
    

    //  //load plugin
    // PluginManager::load(None)?;


    // config for creating a VM
    let config = ConfigBuilder::new(CommonConfigOptions::default())
        .with_host_registration_config(HostRegistrationConfigOptions::default().wasi(true))
        .build()?;
    assert!(config.wasi_enabled());


    // load wasm module from file
    // let module = Module::from_file(Some(&config), wasm_file)?;
    let module = Module::from_file(None, wasm_file)?;


    // create vm without checking wasinn
    let vm = VmBuilder::new()
    .with_config(config)
    .build()?
    .register_module(None, module)?;

    // create a Vm, with wasi_nn plugin, douplicated with another on below
    // let vm = VmBuilder::new()
    //     .with_config(config)
    //     .with_plugin_wasi_nn()
    //     .build()?
    //     .register_module(Some("infer_lib"), module)?;


    // VmDock return a new one, in order to pass complex data such as vec<u8> refering the wasmedge-bindgen example
    let vm = VmDock::new(vm);



    let params = image_tensor_data;
    // check the data type of params
    let params = vec![Param::VecU8(&params)];


    match vm.run_func("tf_infer", params)? {
        Ok(mut res) => {
            println!(
                "Run bindgen -- tf_infer: {:?}",
                res.pop().unwrap().downcast::<Vec<u8>>().unwrap()
            );
        }
        Err(err) => {
            println!("Run bindgen -- tf_infer FAILED {}", err);
        }
    }

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
