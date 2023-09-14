use std::{fs::File, io::Read};
use image;
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
use wasmedge_sdk::{
    config::{CommonConfigOptions, ConfigBuilder, HostRegistrationConfigOptions},
    dock::{Param,VmDock},
    plugin::PluginManager,
    Module, VmBuilder,
    params,
};

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    // infer()?;




    let wasm_file = "rust_mobilenet_food_lib.wasm";
   

    // load wasinn-pytorch-plugin from the default plugin directory: /usr/local/lib/wasmedge
    //as well as other plugins?
    PluginManager::load(None)?;



    let config = ConfigBuilder::new(CommonConfigOptions::default())
        .with_host_registration_config(HostRegistrationConfigOptions::default().wasi(true))
        .build()?;
    assert!(config.wasi_enabled());
    // assert!(config.wasi_nn_enabled());
    
    // load wasm module from file
    let module = Module::from_file(Some(&config), wasm_file)?;

    // create a Vm
    let mut vm = VmBuilder::new()
        .with_config(config)
        .with_plugin_wasi_nn()
        .build()?
        .register_module(Some("extern"), module)?;
    
    // init wasi module
    vm.wasi_module_mut()
        // .ok_or("Not found wasi module")?
        .expect("Not found wasi module");



// VmDock return a new one, 
    let vm = VmDock::new(vm);

    
    //image is a vec<u8>
    let image_name = "./banana.jpg";

    // init wasi module
    // vm.wasi_module_mut();

    let image_tensor_data = image_to_tensor(image_name.to_string(), 192, 192);

    // vm.wasi_module_mut()
    // // .ok_or("Not found wasi module")?
    // .expect("Not found wasi module")
    // .initialize(
    //     Some(vec![wasm_file, image_name]),
    //     None,
    //     Some(vec![dir_mapping]),
    // );



    let params = image_tensor_data;
    let params = vec![Param::VecU8(&params)];

    match vm.run_func("infer", params)? {
        Ok(mut res) => {
            println!(
                "Run bindgen -- infer: {:?}",
                res.pop().unwrap().downcast::<Vec<u8>>().unwrap()
            );
        }
        Err(err) => {
            println!("Run bindgen -- infer FAILED {}", err);
        }
    }

    
    // let params = image_tensor_data;
    // vm.run_func("infer",params)?;

    // let result = vm.run_func_from_module(module, "infer",image_tensor_data)?;

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

// #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
// fn infer() -> Result<(), Box<dyn std::error::Error>> {
//     // parse arguments
//     let args: Vec<String> = std::env::args().collect();
//     let dir_mapping = &args[1];
//     let wasm_file = &args[2];
//     let model_bin = &args[3];
//     let image_file = &args[4];

//     println!("load plugin");

//     // load wasinn-pytorch-plugin from the default plugin directory: /usr/local/lib/wasmedge
//     PluginManager::load(None)?;

//     let config = ConfigBuilder::new(CommonConfigOptions::default())
//         .with_host_registration_config(HostRegistrationConfigOptions::default().wasi(true))
//         .build()?;
//     assert!(config.wasi_enabled());
//     // assert!(config.wasi_nn_enabled());

//     // load wasm module from file
//     let module = Module::from_file(Some(&config), wasm_file)?;

//     // create a Vm
//     let mut vm = VmBuilder::new()
//         .with_config(config)
//         .with_plugin_wasi_nn()
//         .build()?
//         .register_module(Some("extern"), module)?;

//     // init wasi module
//     // vm.wasi_module_mut()
//     //     // .ok_or("Not found wasi module")?
//     //     .expect("Not found wasi module")
//     //     .initialize(
//     //         Some(vec![wasm_file, model_bin, image_file]),
//     //         None,
//     //         Some(vec![dir_mapping]),
//         // );

//     vm.run_func(Some("extern"), "infer", params!())?;

//     Ok(())
// }