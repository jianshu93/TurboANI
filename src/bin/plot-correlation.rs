use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::Parser;
use plotters::prelude::*;

#[derive(Debug, Parser)]
struct Cli {
    #[arg(long)]
    fastani: PathBuf,
    #[arg(long)]
    rust: PathBuf,
    #[arg(long)]
    output_pdf: PathBuf,
    #[arg(long)]
    summary_tsv: PathBuf,
    #[arg(long)]
    joined_tsv: PathBuf,
}

#[derive(Debug, Clone)]
struct AniRow {
    query: String,
    reference: String,
    ani: f64,
}

#[derive(Debug, Clone)]
struct JoinedPair {
    query: String,
    reference: String,
    fastani: f64,
    rust: f64,
    delta: f64,
}

#[derive(Debug, Clone)]
struct Summary {
    fastani_rows: usize,
    rust_rows: usize,
    common_pairs: usize,
    fastani_only: usize,
    rust_only: usize,
    self_pairs: usize,
    pearson: f64,
    spearman: f64,
    mean_abs_delta: f64,
    median_abs_delta: f64,
    rmse: f64,
    max_abs_delta: f64,
    min_fastani: f64,
    max_fastani: f64,
    min_rust: f64,
    max_rust: f64,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let fastani = read_ani_table(&cli.fastani)?;
    let rust = read_ani_table(&cli.rust)?;
    let (joined, summary) = join_and_summarize(&fastani, &rust)?;
    write_joined(&cli.joined_tsv, &joined)?;
    write_summary(&cli.summary_tsv, &summary)?;
    draw_plot(&cli.output_pdf, &joined, &summary)?;
    Ok(())
}

fn read_ani_table(path: &Path) -> Result<HashMap<(String, String), AniRow>> {
    let text =
        fs::read_to_string(path).with_context(|| format!("failed to read {}", path.display()))?;
    let mut rows = HashMap::new();
    for (line_no, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let fields = line.split_whitespace().collect::<Vec<_>>();
        anyhow::ensure!(
            fields.len() >= 3,
            "expected at least 3 columns in {} line {}",
            path.display(),
            line_no + 1
        );
        let query = fields[0].to_string();
        let reference = fields[1].to_string();
        let ani = fields[2].parse::<f64>().with_context(|| {
            format!(
                "failed to parse ANI in {} line {}",
                path.display(),
                line_no + 1
            )
        })?;
        rows.insert(
            (query.clone(), reference.clone()),
            AniRow {
                query,
                reference,
                ani,
            },
        );
    }
    Ok(rows)
}

fn join_and_summarize(
    fastani: &HashMap<(String, String), AniRow>,
    rust: &HashMap<(String, String), AniRow>,
) -> Result<(Vec<JoinedPair>, Summary)> {
    let mut joined = Vec::new();
    for (key, fast_row) in fastani {
        if let Some(rust_row) = rust.get(key) {
            joined.push(JoinedPair {
                query: fast_row.query.clone(),
                reference: fast_row.reference.clone(),
                fastani: fast_row.ani,
                rust: rust_row.ani,
                delta: rust_row.ani - fast_row.ani,
            });
        }
    }
    joined.sort_by(|a, b| {
        a.query
            .cmp(&b.query)
            .then_with(|| a.reference.cmp(&b.reference))
    });
    anyhow::ensure!(!joined.is_empty(), "no shared ANI pairs found");

    let fast_keys = fastani.keys().collect::<HashSet<_>>();
    let rust_keys = rust.keys().collect::<HashSet<_>>();
    let fastani_only = fast_keys.difference(&rust_keys).count();
    let rust_only = rust_keys.difference(&fast_keys).count();
    let self_pairs = joined
        .iter()
        .filter(|pair| pair.query == pair.reference)
        .count();
    let fast_values = joined.iter().map(|pair| pair.fastani).collect::<Vec<_>>();
    let rust_values = joined.iter().map(|pair| pair.rust).collect::<Vec<_>>();
    let abs_delta = joined
        .iter()
        .map(|pair| pair.delta.abs())
        .collect::<Vec<_>>();

    let summary = Summary {
        fastani_rows: fastani.len(),
        rust_rows: rust.len(),
        common_pairs: joined.len(),
        fastani_only,
        rust_only,
        self_pairs,
        pearson: pearson(&fast_values, &rust_values),
        spearman: spearman(&fast_values, &rust_values),
        mean_abs_delta: mean(&abs_delta),
        median_abs_delta: median(abs_delta.clone()),
        rmse: (joined
            .iter()
            .map(|pair| pair.delta * pair.delta)
            .sum::<f64>()
            / joined.len() as f64)
            .sqrt(),
        max_abs_delta: abs_delta.into_iter().fold(0.0f64, f64::max),
        min_fastani: fast_values.iter().copied().fold(f64::INFINITY, f64::min),
        max_fastani: fast_values
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max),
        min_rust: rust_values.iter().copied().fold(f64::INFINITY, f64::min),
        max_rust: rust_values
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max),
    };
    Ok((joined, summary))
}

fn write_joined(path: &Path, joined: &[JoinedPair]) -> Result<()> {
    ensure_parent(path)?;
    let mut out = BufWriter::new(
        fs::File::create(path).with_context(|| format!("failed to create {}", path.display()))?,
    );
    writeln!(
        out,
        "query\treference\tfastani_ani\trust_fastani_ani\tdelta\tabs_delta"
    )?;
    for pair in joined {
        writeln!(
            out,
            "{}\t{}\t{:.6}\t{:.6}\t{:.6}\t{:.6}",
            pair.query,
            pair.reference,
            pair.fastani,
            pair.rust,
            pair.delta,
            pair.delta.abs()
        )?;
    }
    Ok(())
}

fn write_summary(path: &Path, summary: &Summary) -> Result<()> {
    ensure_parent(path)?;
    let mut out = BufWriter::new(
        fs::File::create(path).with_context(|| format!("failed to create {}", path.display()))?,
    );
    writeln!(out, "metric\tvalue")?;
    writeln!(out, "fastani_rows\t{}", summary.fastani_rows)?;
    writeln!(out, "rust_rows\t{}", summary.rust_rows)?;
    writeln!(out, "common_pairs\t{}", summary.common_pairs)?;
    writeln!(out, "fastani_only\t{}", summary.fastani_only)?;
    writeln!(out, "rust_only\t{}", summary.rust_only)?;
    writeln!(out, "self_pairs\t{}", summary.self_pairs)?;
    writeln!(out, "pearson\t{:.10}", summary.pearson)?;
    writeln!(out, "spearman\t{:.10}", summary.spearman)?;
    writeln!(out, "mean_abs_delta\t{:.6}", summary.mean_abs_delta)?;
    writeln!(out, "median_abs_delta\t{:.6}", summary.median_abs_delta)?;
    writeln!(out, "rmse\t{:.6}", summary.rmse)?;
    writeln!(out, "max_abs_delta\t{:.6}", summary.max_abs_delta)?;
    writeln!(out, "min_fastani\t{:.6}", summary.min_fastani)?;
    writeln!(out, "max_fastani\t{:.6}", summary.max_fastani)?;
    writeln!(out, "min_rust\t{:.6}", summary.min_rust)?;
    writeln!(out, "max_rust\t{:.6}", summary.max_rust)?;
    Ok(())
}

fn draw_plot(output_pdf: &Path, joined: &[JoinedPair], summary: &Summary) -> Result<()> {
    ensure_parent(output_pdf)?;
    let output_svg = output_pdf.with_extension("svg");
    let min_axis = summary.min_fastani.min(summary.min_rust).floor().max(75.0);
    let max_axis = summary.max_fastani.max(summary.max_rust).ceil().min(100.5);
    let legend_max = summary.max_abs_delta.max(0.5);

    let mut svg = String::new();
    {
        let root = SVGBackend::with_string(&mut svg, (1100, 900)).into_drawing_area();
        root.fill(&WHITE)
            .map_err(|e| anyhow::anyhow!("failed to initialize SVG drawing area: {e:?}"))?;
        let (plot_area, legend_area) = root.split_horizontally(930);
        let mut chart = ChartBuilder::on(&plot_area)
            .caption(
                format!(
                    "Streptomycetaceae 60-genome ANI correlation: n={}, Pearson={:.5}, Spearman={:.5}",
                    summary.common_pairs, summary.pearson, summary.spearman
                ),
                ("sans-serif", 20),
            )
            .margin(26)
            .x_label_area_size(54)
            .y_label_area_size(64)
            .build_cartesian_2d(min_axis..max_axis, min_axis..max_axis)
            .map_err(|e| anyhow::anyhow!("failed to build correlation chart: {e:?}"))?;
        chart
            .configure_mesh()
            .x_desc("Original fastANI ANI (%)")
            .y_desc("Rust rust-fastani ANI (%)")
            .light_line_style(RGBColor(232, 232, 232))
            .draw()
            .map_err(|e| anyhow::anyhow!("failed to draw chart mesh: {e:?}"))?;
        chart
            .draw_series(LineSeries::new(
                vec![(min_axis, min_axis), (max_axis, max_axis)],
                BLACK.mix(0.45).stroke_width(2),
            ))
            .map_err(|e| anyhow::anyhow!("failed to draw identity line: {e:?}"))?;
        chart
            .draw_series(joined.iter().map(|pair| {
                Circle::new(
                    (pair.fastani, pair.rust),
                    2,
                    delta_color(pair.delta.abs(), legend_max).mix(0.62).filled(),
                )
            }))
            .map_err(|e| anyhow::anyhow!("failed to draw correlation points: {e:?}"))?;
        chart
            .draw_series([Text::new(
                format!(
                    "mean |delta|={:.3}%, median={:.3}%, max={:.3}%",
                    summary.mean_abs_delta, summary.median_abs_delta, summary.max_abs_delta
                ),
                (min_axis + 0.4, max_axis - 0.9),
                ("sans-serif", 15).into_font(),
            )])
            .map_err(|e| anyhow::anyhow!("failed to draw chart annotation: {e:?}"))?;

        draw_delta_colorbar(&legend_area, legend_max)?;
        root.present()
            .map_err(|e| anyhow::anyhow!("failed to finalize SVG plot: {e:?}"))?;
    }

    fs::write(&output_svg, &svg)
        .with_context(|| format!("failed to write SVG {}", output_svg.display()))?;
    let mut options = svg2pdf::usvg::Options::default();
    options.fontdb_mut().load_system_fonts();
    let tree = svg2pdf::usvg::Tree::from_str(&svg, &options).map_err(|e| {
        anyhow::anyhow!("failed to parse correlation SVG before PDF conversion: {e}")
    })?;
    let pdf = svg2pdf::to_pdf(
        &tree,
        svg2pdf::ConversionOptions::default(),
        svg2pdf::PageOptions::default(),
    )
    .map_err(|e| anyhow::anyhow!("failed to convert correlation SVG to PDF: {e}"))?;
    fs::write(output_pdf, pdf)
        .with_context(|| format!("failed to write PDF {}", output_pdf.display()))?;
    Ok(())
}

fn draw_delta_colorbar(
    area: &DrawingArea<SVGBackend<'_>, plotters::coord::Shift>,
    legend_max: f64,
) -> Result<()> {
    area.fill(&WHITE)
        .map_err(|e| anyhow::anyhow!("failed to initialize legend area: {e:?}"))?;
    let x0 = 36i32;
    let y0 = 145i32;
    let bar_width = 26i32;
    let bar_height = 360i32;
    area.draw(&Text::new(
        "abs delta",
        (x0 - 8, y0 - 50),
        ("sans-serif", 15).into_font(),
    ))
    .map_err(|e| anyhow::anyhow!("failed to draw delta legend title: {e:?}"))?;
    area.draw(&Text::new(
        "(ANI %)",
        (x0 - 8, y0 - 30),
        ("sans-serif", 13).into_font(),
    ))
    .map_err(|e| anyhow::anyhow!("failed to draw delta legend unit: {e:?}"))?;

    for pixel in 0..bar_height {
        let fraction = 1.0 - pixel as f64 / (bar_height - 1) as f64;
        let delta = legend_max * fraction;
        area.draw(&Rectangle::new(
            [(x0, y0 + pixel), (x0 + bar_width, y0 + pixel + 1)],
            delta_color(delta, legend_max).filled(),
        ))
        .map_err(|e| anyhow::anyhow!("failed to draw delta colorbar: {e:?}"))?;
    }
    area.draw(&Rectangle::new(
        [(x0, y0), (x0 + bar_width, y0 + bar_height)],
        BLACK.stroke_width(1),
    ))
    .map_err(|e| anyhow::anyhow!("failed to draw delta colorbar frame: {e:?}"))?;

    for (fraction, label) in [
        (0.0, format!("{:.2}", legend_max)),
        (0.5, format!("{:.2}", legend_max / 2.0)),
        (1.0, "0.00".to_string()),
    ] {
        let y = y0 + (fraction * bar_height as f64).round() as i32;
        area.draw(&PathElement::new(
            vec![(x0 + bar_width, y), (x0 + bar_width + 6, y)],
            BLACK,
        ))
        .map_err(|e| anyhow::anyhow!("failed to draw delta colorbar tick: {e:?}"))?;
        area.draw(&Text::new(
            label,
            (x0 + bar_width + 12, y - 7),
            ("sans-serif", 12).into_font(),
        ))
        .map_err(|e| anyhow::anyhow!("failed to draw delta colorbar label: {e:?}"))?;
    }
    Ok(())
}

fn delta_color(delta: f64, max_delta: f64) -> HSLColor {
    let t = if max_delta > 0.0 {
        (delta / max_delta).clamp(0.0, 1.0)
    } else {
        0.0
    };
    HSLColor((210.0 - 210.0 * t) / 360.0, 0.78, 0.45)
}

fn ensure_parent(path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed to create {}", parent.display()))?;
        }
    }
    Ok(())
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(|a, b| a.total_cmp(b));
    let mid = values.len() / 2;
    if values.len() % 2 == 0 {
        (values[mid - 1] + values[mid]) / 2.0
    } else {
        values[mid]
    }
}

fn pearson(x: &[f64], y: &[f64]) -> f64 {
    let x_mean = mean(x);
    let y_mean = mean(y);
    let mut covariance = 0.0;
    let mut x_var = 0.0;
    let mut y_var = 0.0;
    for (&xv, &yv) in x.iter().zip(y) {
        let dx = xv - x_mean;
        let dy = yv - y_mean;
        covariance += dx * dy;
        x_var += dx * dx;
        y_var += dy * dy;
    }
    covariance / (x_var.sqrt() * y_var.sqrt())
}

fn spearman(x: &[f64], y: &[f64]) -> f64 {
    pearson(&ranks(x), &ranks(y))
}

fn ranks(values: &[f64]) -> Vec<f64> {
    let mut indexed = values
        .iter()
        .copied()
        .enumerate()
        .collect::<Vec<(usize, f64)>>();
    indexed.sort_by(|a, b| a.1.total_cmp(&b.1));
    let mut ranks = vec![0.0; values.len()];
    let mut i = 0usize;
    while i < indexed.len() {
        let start = i;
        let value = indexed[i].1;
        while i < indexed.len() && indexed[i].1 == value {
            i += 1;
        }
        let rank = (start + 1 + i) as f64 / 2.0;
        for &(original_index, _) in &indexed[start..i] {
            ranks[original_index] = rank;
        }
    }
    ranks
}
