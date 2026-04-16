use anyhow::{bail, Context, Result};
use serde_json::json;
use std::{fs, path::PathBuf};

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let mut input: Option<PathBuf> = None;
    let mut report = PathBuf::from("xla_hlo/reports/hlo_report.json");
    let mut validate_only = false;
    let mut run = false;
    let mut kernel_dir = PathBuf::from("kernels");
    let mut device = String::from("any");
    let mut input_f32: Option<PathBuf> = None;
    let mut input1_f32: Option<PathBuf> = None;
    let mut inputs_f32: Vec<PathBuf> = Vec::new();
    let mut output_f32: Option<PathBuf> = None;
    let mut expected_f32: Option<PathBuf> = None;
    let mut epsilon = 1.0e-6_f32;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--input" => input = Some(PathBuf::from(args.next().context("--input requires path")?)),
            "--report" => report = PathBuf::from(args.next().context("--report requires path")?),
            "--validate-only" => validate_only = true,
            "--run" => run = true,
            "--kernel-dir" => {
                kernel_dir = PathBuf::from(args.next().context("--kernel-dir requires path")?)
            }
            "--device" => device = args.next().context("--device requires selector")?,
            "--input-f32" => {
                input_f32 = Some(PathBuf::from(
                    args.next().context("--input-f32 requires path")?,
                ))
            }
            "--inputs-f32" => {
                inputs_f32 = args
                    .next()
                    .context("--inputs-f32 requires comma-separated paths")?
                    .split(',')
                    .filter(|s| !s.is_empty())
                    .map(PathBuf::from)
                    .collect();
            }
            "--input1-f32" => {
                input1_f32 = Some(PathBuf::from(
                    args.next().context("--input1-f32 requires path")?,
                ))
            }
            "--output-f32" => {
                output_f32 = Some(PathBuf::from(
                    args.next().context("--output-f32 requires path")?,
                ))
            }
            "--expected-f32" => {
                expected_f32 = Some(PathBuf::from(
                    args.next().context("--expected-f32 requires path")?,
                ))
            }
            "--epsilon" => {
                epsilon = args
                    .next()
                    .context("--epsilon requires value")?
                    .parse()
                    .context("parsing --epsilon as f32")?
            }
            other => bail!("unknown argument {other}"),
        }
    }

    let input = input.context("--input is required")?;
    let text =
        fs::read_to_string(&input).with_context(|| format!("reading {}", input.display()))?;
    let module = xla_hlo::parser::parse_stablehlo_text(&text)?;
    let hist = xla_hlo::compiler::op_histogram(&module);
    let validation = match xla_hlo::compiler::validate_supported(&module) {
        Ok(()) => json!({"status":"pass"}),
        Err(e) => json!({"status":"blocked_unsupported_feature", "reason": e.to_string()}),
    };
    let execution = if run {
        if inputs_f32.is_empty() {
            let input0 = input_f32
                .as_deref()
                .context("--run requires --input-f32 or --inputs-f32")?;
            inputs_f32.push(input0.to_path_buf());
            if let Some(input1) = input1_f32.as_deref() {
                inputs_f32.push(input1.to_path_buf());
            }
        }
        let input_refs = inputs_f32.iter().map(PathBuf::as_path).collect::<Vec<_>>();
        xla_hlo::compiler::run_graph(
            &module,
            &kernel_dir,
            &device,
            &input_refs,
            output_f32.as_deref(),
            expected_f32.as_deref(),
            epsilon,
        )?
    } else {
        json!({"status":"not_requested"})
    };
    let out = json!({
        "status": if execution["status"] == "fail" {
            json!("fail")
        } else if execution["status"] == "pass" {
            json!("pass")
        } else {
            validation["status"].clone()
        },
        "input": input,
        "module": module,
        "op_histogram": hist,
        "validation": validation,
        "execution": execution,
        "validate_only": validate_only
    });
    xla_hlo::report::write_json(&report, &out)?;
    println!("{}", serde_json::to_string_pretty(&out)?);
    Ok(())
}
