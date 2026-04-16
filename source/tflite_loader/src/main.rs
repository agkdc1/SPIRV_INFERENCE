use anyhow::{bail, Context, Result};
use serde_json::json;
use std::path::PathBuf;
use tflite_loader::{engine, graph, model, report};

#[derive(Debug)]
struct Args {
    model: Option<PathBuf>,
    input_f32: Option<PathBuf>,
    kernel_dir: PathBuf,
    report: PathBuf,
    dump_dir: PathBuf,
    device: String,
    self_test_kernels: bool,
    inspect_only: bool,
}

fn main() -> Result<()> {
    let args = parse_args()?;
    let mut sections = serde_json::Map::new();
    let mut final_status = "pass";

    if args.self_test_kernels {
        let r = engine::run_kernel_self_test(&args.kernel_dir, &args.device, &args.dump_dir)
            .context("running Vulkan SPIR-V kernel self-test")?;
        if r.get("status").and_then(|v| v.as_str()) != Some("pass") {
            final_status = "fail";
        }
        sections.insert("kernel_self_test".to_string(), r);
    }

    if let Some(model_path) = &args.model {
        let info = model::load_model(model_path)?;
        let graph_status = match graph::validate_supported_graph(&info) {
            Ok(v) => v,
            Err(e) => {
                final_status = "blocked_unsupported_feature";
                json!({"status": "blocked_unsupported_feature", "reason": e.to_string()})
            }
        };
        sections.insert("model".to_string(), serde_json::to_value(&info)?);
        sections.insert("graph_validation".to_string(), graph_status);
        if !args.inspect_only && final_status == "pass" {
            match engine::run_model(
                &info,
                &args.kernel_dir,
                &args.device,
                &args.dump_dir,
                args.input_f32.as_deref(),
            ) {
                Ok(v) => {
                    if v.get("status").and_then(|s| s.as_str()) != Some("pass") {
                        final_status = "fail";
                    }
                    sections.insert("execution".to_string(), v);
                }
                Err(e) => {
                    final_status = "blocked_unsupported_feature";
                    sections.insert(
                        "execution".to_string(),
                        json!({
                            "status": "blocked_unsupported_feature",
                            "reason": e.to_string()
                        }),
                    );
                }
            }
        }
    }

    if !args.self_test_kernels && args.model.is_none() {
        bail!("nothing to do; pass --self-test-kernels and/or --model <path>");
    }

    let out = json!({
        "status": final_status,
        "crate": "tflite_loader",
        "kernel_dir": args.kernel_dir,
        "input_f32": args.input_f32,
        "dump_dir": args.dump_dir,
        "sections": sections,
        "non_claims": {
            "uses_cuda": false,
            "uses_cublas": false,
            "uses_cudnn": false,
            "claims_full_tensorflow_runtime": false
        }
    });
    report::write_json_report(&args.report, &out)?;
    println!("{}", serde_json::to_string_pretty(&out)?);
    Ok(())
}

fn parse_args() -> Result<Args> {
    let mut args = std::env::args().skip(1);
    let mut out = Args {
        model: None,
        input_f32: None,
        kernel_dir: PathBuf::from("kernels"),
        report: PathBuf::from("tflite_loader/out/tflite_report.json"),
        dump_dir: PathBuf::from("tflite_loader/out/tensors"),
        device: "any".to_string(),
        self_test_kernels: false,
        inspect_only: false,
    };
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--model" => {
                out.model = Some(PathBuf::from(
                    args.next().context("--model requires a path")?,
                ))
            }
            "--input-f32" => {
                out.input_f32 = Some(PathBuf::from(
                    args.next().context("--input-f32 requires a path")?,
                ))
            }
            "--kernel-dir" => {
                out.kernel_dir = PathBuf::from(args.next().context("--kernel-dir requires a path")?)
            }
            "--report" => {
                out.report = PathBuf::from(args.next().context("--report requires a path")?)
            }
            "--dump-dir" => {
                out.dump_dir = PathBuf::from(args.next().context("--dump-dir requires a path")?)
            }
            "--device" => out.device = args.next().context("--device requires intel|nvidia|any")?,
            "--self-test-kernels" => out.self_test_kernels = true,
            "--inspect-only" => out.inspect_only = true,
            "--help" | "-h" => {
                println!("usage: tflite_loader [--self-test-kernels] [--model MODEL.tflite] [--input-f32 input.raw.f32] [--inspect-only] [--kernel-dir kernels] [--device any|intel|nvidia] --report out.json --dump-dir dir");
                std::process::exit(0);
            }
            other => bail!("unknown argument {other}"),
        }
    }
    Ok(out)
}
