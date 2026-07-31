use std::fmt::Write as FmtWrite;
use std::fs;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use indicatif::ProgressBar;
use plotters::coord::Shift;
use plotters::prelude::*;

use crate::compute_identity::{DistanceTableCache, cmp_f64, compute_ani_results};
use crate::simd_minimizer::deterministic_tab64_twisted;
use crate::sliding_mapper::MappingResult;
use crate::{AniConfig, AniResult, QueryFileData, ReferenceIndex, map_query_file, read_query_file};

pub fn write_pair_visualization_pdf(
    query_path: impl AsRef<Path>,
    ref_path: impl AsRef<Path>,
    config: &AniConfig,
    output_path: impl AsRef<Path>,
) -> Result<Vec<AniResult>> {
    config.validate()?;
    let query_path = query_path.as_ref();
    let ref_path = ref_path.as_ref();
    let output_path = output_path.as_ref();
    let window_size = config.resolved_window_size();
    let tab_hasher = deterministic_tab64_twisted(config.tab_hash_seed);
    let ref_paths = vec![ref_path.to_path_buf()];
    let reference_progress = ProgressBar::hidden();
    let (reference, _) = ReferenceIndex::build(
        &ref_paths,
        config,
        window_size,
        &tab_hasher,
        &reference_progress,
    )?;
    let distance_cache =
        DistanceTableCache::new(config.kmer_size, config.fragment_len, config.distance_model);
    let query = read_query_file(query_path, config)?;
    let (mappings, _) = map_query_file(&query, &reference, config, window_size, &distance_cache)?;

    // The raw L2 mappings can contain several candidate target regions for one
    // query fragment. The ANI calculation subsequently keeps one best mapping
    // per query fragment/reference genome and then one best query mapping per
    // reference-position bin. Use the same two-way filtered set for plotting so
    // the conserved view matches the mappings that contribute to ANI instead of
    // displaying every raw L2 candidate hit.
    let visualization_mappings = select_best_visualization_mappings(&mappings, &reference, config);
    let output_paths = visualization_output_paths(output_path);
    write_visualization_map_data(
        &output_paths.map_data,
        query_path,
        ref_path,
        &query,
        &reference,
        &visualization_mappings,
    )?;
    let points = visualization_points(&query, &reference, &visualization_mappings);
    let results = compute_ani_results(&query, &reference, mappings, config);
    draw_pair_visualization_pdf(
        &output_paths,
        query_path,
        ref_path,
        &query,
        &reference,
        &results,
        &points,
    )?;
    Ok(results)
}

#[derive(Debug, Clone)]
struct VisualizationPoint {
    query_start: f64,
    query_end: f64,
    ref_start: f64,
    ref_end: f64,
    identity: f64,
}

impl VisualizationPoint {
    fn query_mid(&self) -> f64 {
        (self.query_start + self.query_end) / 2.0
    }

    fn ref_mid(&self) -> f64 {
        (self.ref_start + self.ref_end) / 2.0
    }
}

fn visualization_points(
    query: &QueryFileData,
    reference: &ReferenceIndex,
    mappings: &[MappingResult],
) -> Vec<VisualizationPoint> {
    let ref_offsets = reference_contig_offsets(reference, 0);
    let mut points = mappings
        .iter()
        .filter_map(|mapping| {
            if reference.contigs[mapping.ref_seq_id].genome_id != 0 {
                return None;
            }
            let query_fragment = query.fragments.get(mapping.query_seq_id)?;
            let ref_offset = *ref_offsets.get(mapping.ref_seq_id)?;
            let query_start = query_fragment.global_start as f64;
            let ref_start = ref_offset as f64 + mapping.ref_start as f64;
            Some(VisualizationPoint {
                query_start,
                query_end: query_start + mapping.query_len as f64,
                ref_start,
                ref_end: ref_offset as f64 + (mapping.ref_end + 1) as f64,
                identity: mapping.identity,
            })
        })
        .collect::<Vec<_>>();
    points.sort_by(|a, b| {
        cmp_f64(a.query_start, b.query_start).then_with(|| cmp_f64(a.ref_start, b.ref_start))
    });
    points
}

fn select_best_visualization_mappings(
    mappings: &[MappingResult],
    reference: &ReferenceIndex,
    config: &AniConfig,
) -> Vec<MappingResult> {
    #[derive(Debug, Clone, Copy)]
    struct IndexedVisualMapping {
        mapping_index: usize,
        ref_seq_id: usize,
        genome_id: usize,
        query_seq_id: usize,
        ref_start: usize,
        map_ref_pos_bin: usize,
        identity: f64,
    }

    let mut one_way_candidates = mappings
        .iter()
        .enumerate()
        .map(|(mapping_index, mapping)| IndexedVisualMapping {
            mapping_index,
            ref_seq_id: mapping.ref_seq_id,
            genome_id: reference.contigs[mapping.ref_seq_id].genome_id,
            query_seq_id: mapping.query_seq_id,
            ref_start: mapping.ref_start,
            map_ref_pos_bin: mapping.ref_start / config.fragment_len.saturating_sub(20).max(1),
            identity: mapping.identity,
        })
        .collect::<Vec<_>>();

    // This is the same ordering and "keep last" rule used by
    // compute_ani_results(): the last record in each genome/query-fragment group
    // is the best identity, with reference contig and position as tie breakers.
    one_way_candidates.sort_unstable_by(|a, b| {
        (a.genome_id, a.query_seq_id)
            .cmp(&(b.genome_id, b.query_seq_id))
            .then_with(|| cmp_f64(a.identity, b.identity))
            .then_with(|| a.ref_seq_id.cmp(&b.ref_seq_id))
            .then_with(|| a.ref_start.cmp(&b.ref_start))
    });

    let mut one_way = Vec::<IndexedVisualMapping>::new();
    for mapping in one_way_candidates {
        if let Some(last) = one_way.last_mut() {
            if last.genome_id == mapping.genome_id && last.query_seq_id == mapping.query_seq_id {
                *last = mapping;
                continue;
            }
        }
        one_way.push(mapping);
    }

    // Apply the same reciprocal/reference-bin uniqueness filter used by the ANI
    // reducer. This prevents many query fragments from being drawn onto the same
    // target region and makes the plotted mapped-fragment count correspond to the
    // two-way set that contributes to ANI.
    one_way.sort_unstable_by(|a, b| {
        (a.ref_seq_id, a.map_ref_pos_bin)
            .cmp(&(b.ref_seq_id, b.map_ref_pos_bin))
            .then_with(|| cmp_f64(a.identity, b.identity))
    });

    let mut two_way = Vec::<IndexedVisualMapping>::new();
    for mapping in one_way {
        if let Some(last) = two_way.last_mut() {
            if last.ref_seq_id == mapping.ref_seq_id
                && last.map_ref_pos_bin == mapping.map_ref_pos_bin
            {
                *last = mapping;
                continue;
            }
        }
        two_way.push(mapping);
    }

    two_way
        .into_iter()
        .map(|mapping| mappings[mapping.mapping_index].clone())
        .collect()
}

fn write_visualization_map_data(
    output_path: &Path,
    query_path: &Path,
    ref_path: &Path,
    query: &QueryFileData,
    reference: &ReferenceIndex,
    mappings: &[MappingResult],
) -> Result<()> {
    ensure_parent_dir(output_path)?;
    let ref_offsets = reference_contig_offsets(reference, 0);
    let file = fs::File::create(output_path).with_context(|| {
        format!(
            "failed to create visualization map {}",
            output_path.display()
        )
    })?;
    let mut writer = BufWriter::new(file);

    writeln!(
        writer,
        "query\treference\tquery_fragment\tquery_start\tquery_end\treference_contig\treference_contig_start\treference_contig_end\treference_start\treference_end\tidentity\tidentity_upper_bound\tconserved_sketches\tsketch_size"
    )
    .with_context(|| format!("failed to write visualization map {}", output_path.display()))?;

    for mapping in mappings {
        if reference.contigs[mapping.ref_seq_id].genome_id != 0 {
            continue;
        }
        let Some(query_fragment) = query.fragments.get(mapping.query_seq_id) else {
            continue;
        };
        let Some(ref_offset) = ref_offsets.get(mapping.ref_seq_id) else {
            continue;
        };

        let ref_contig = &reference.contigs[mapping.ref_seq_id];
        let query_start = query_fragment.global_start;
        let query_end = query_start + mapping.query_len;
        let ref_start = *ref_offset + mapping.ref_start;
        let ref_end = *ref_offset + mapping.ref_end + 1;

        writeln!(
            writer,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.6}\t{:.6}\t{}\t{}",
            display_name(query_path),
            display_name(ref_path),
            mapping.query_seq_id,
            query_start,
            query_end,
            ref_contig.name,
            mapping.ref_start,
            mapping.ref_end + 1,
            ref_start,
            ref_end,
            mapping.identity,
            mapping.identity_upper_bound,
            mapping.conserved_sketches,
            mapping.sketch_size,
        )
        .with_context(|| {
            format!(
                "failed to write visualization map {}",
                output_path.display()
            )
        })?;
    }

    writer.flush().with_context(|| {
        format!(
            "failed to flush visualization map {}",
            output_path.display()
        )
    })?;
    Ok(())
}

fn reference_contig_offsets(reference: &ReferenceIndex, genome_id: usize) -> Vec<usize> {
    let mut offsets = vec![0usize; reference.contigs.len()];
    let mut offset = 0usize;
    for (seq_id, contig) in reference.contigs.iter().enumerate() {
        if contig.genome_id == genome_id {
            offsets[seq_id] = offset;
            offset += contig.len;
        }
    }
    offsets
}

fn draw_pair_visualization_pdf(
    paths: &VisualizationOutputPaths,
    query_path: &Path,
    ref_path: &Path,
    query: &QueryFileData,
    reference: &ReferenceIndex,
    results: &[AniResult],
    points: &[VisualizationPoint],
) -> Result<()> {
    let query_mbp = (query.genome_len as f64 / 1_000_000.0).max(0.001);
    let ref_mbp = (reference.genomes[0].length as f64 / 1_000_000.0).max(0.001);
    let ani_label = ani_label(results);

    let combined_svg = render_combined_visualization_svg(
        query_path, ref_path, query_mbp, ref_mbp, &ani_label, results, points,
    )?;
    write_vector_plot(&combined_svg, &paths.combined_svg, &paths.combined_pdf)?;

    let map_svg =
        render_map_visualization_svg(query_path, ref_path, query_mbp, ref_mbp, &ani_label, points)?;
    write_vector_plot(&map_svg, &paths.map_svg, &paths.map_pdf)?;

    let identity_svg = render_identity_visualization_svg(query_mbp, results, points)?;
    write_vector_plot(&identity_svg, &paths.identity_svg, &paths.identity_pdf)?;

    let conserved_svg = render_conserved_visualization_svg(
        query_path, ref_path, query_mbp, ref_mbp, &ani_label, points,
    )?;
    write_vector_plot(&conserved_svg, &paths.conserved_svg, &paths.conserved_pdf)?;

    Ok(())
}

#[derive(Debug, Clone)]
struct VisualizationOutputPaths {
    combined_pdf: PathBuf,
    combined_svg: PathBuf,
    map_data: PathBuf,
    map_pdf: PathBuf,
    map_svg: PathBuf,
    identity_pdf: PathBuf,
    identity_svg: PathBuf,
    conserved_pdf: PathBuf,
    conserved_svg: PathBuf,
}

fn visualization_output_paths(output_path: &Path) -> VisualizationOutputPaths {
    VisualizationOutputPaths {
        combined_pdf: output_path.to_path_buf(),
        combined_svg: output_path.with_extension("svg"),
        map_data: output_path.with_extension("map"),
        map_pdf: visualization_sidecar_path(output_path, "map", "pdf"),
        map_svg: visualization_sidecar_path(output_path, "map", "svg"),
        identity_pdf: visualization_sidecar_path(output_path, "identity", "pdf"),
        identity_svg: visualization_sidecar_path(output_path, "identity", "svg"),
        conserved_pdf: visualization_sidecar_path(output_path, "conserved", "pdf"),
        conserved_svg: visualization_sidecar_path(output_path, "conserved", "svg"),
    }
}

fn visualization_sidecar_path(output_path: &Path, suffix: &str, extension: &str) -> PathBuf {
    let stem = output_path
        .file_stem()
        .and_then(|name| name.to_str())
        .unwrap_or("visualization");
    let mut path = output_path.to_path_buf();
    path.set_file_name(format!("{stem}.{suffix}.{extension}"));
    path
}

fn write_vector_plot(svg: &str, svg_path: &Path, pdf_path: &Path) -> Result<()> {
    ensure_parent_dir(svg_path)?;
    ensure_parent_dir(pdf_path)?;
    fs::write(svg_path, svg)
        .with_context(|| format!("failed to write SVG {}", svg_path.display()))?;

    let mut options = svg2pdf::usvg::Options::default();
    options.fontdb_mut().load_system_fonts();
    let tree = svg2pdf::usvg::Tree::from_str(svg, &options)
        .map_err(|e| anyhow::anyhow!("failed to parse plot SVG before PDF conversion: {e}"))?;
    let pdf = svg2pdf::to_pdf(
        &tree,
        svg2pdf::ConversionOptions::default(),
        svg2pdf::PageOptions::default(),
    )
    .map_err(|e| anyhow::anyhow!("failed to convert SVG plot to PDF: {e}"))?;
    fs::write(pdf_path, pdf)
        .with_context(|| format!("failed to write PDF {}", pdf_path.display()))?;
    Ok(())
}

fn ensure_parent_dir(path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed to create {}", parent.display()))?;
        }
    }
    Ok(())
}

fn render_combined_visualization_svg(
    query_path: &Path,
    ref_path: &Path,
    query_mbp: f64,
    ref_mbp: f64,
    ani_label: &str,
    results: &[AniResult],
    points: &[VisualizationPoint],
) -> Result<String> {
    let mut svg = String::new();
    {
        let root = SVGBackend::with_string(&mut svg, (500, 720)).into_drawing_area();
        root.fill(&WHITE)
            .map_err(|e| anyhow::anyhow!("failed to initialize SVG drawing area: {e:?}"))?;
        let areas = root.split_evenly((2, 1));
        draw_map_chart(
            &areas[0], query_path, ref_path, query_mbp, ref_mbp, ani_label, points,
        )?;
        draw_identity_chart(&areas[1], query_mbp, results, points)?;
        root.present()
            .map_err(|e| anyhow::anyhow!("failed to finalize SVG plot: {e:?}"))?;
    }
    Ok(svg)
}

fn render_map_visualization_svg(
    query_path: &Path,
    ref_path: &Path,
    query_mbp: f64,
    ref_mbp: f64,
    ani_label: &str,
    points: &[VisualizationPoint],
) -> Result<String> {
    let mut svg = String::new();
    {
        let root = SVGBackend::with_string(&mut svg, (500, 360)).into_drawing_area();
        root.fill(&WHITE)
            .map_err(|e| anyhow::anyhow!("failed to initialize map SVG drawing area: {e:?}"))?;
        draw_map_chart(
            &root, query_path, ref_path, query_mbp, ref_mbp, ani_label, points,
        )?;
        root.present()
            .map_err(|e| anyhow::anyhow!("failed to finalize map SVG plot: {e:?}"))?;
    }
    Ok(svg)
}

fn render_identity_visualization_svg(
    query_mbp: f64,
    results: &[AniResult],
    points: &[VisualizationPoint],
) -> Result<String> {
    let mut svg = String::new();
    {
        let root = SVGBackend::with_string(&mut svg, (500, 360)).into_drawing_area();
        root.fill(&WHITE).map_err(|e| {
            anyhow::anyhow!("failed to initialize identity SVG drawing area: {e:?}")
        })?;
        draw_identity_chart(&root, query_mbp, results, points)?;
        root.present()
            .map_err(|e| anyhow::anyhow!("failed to finalize identity SVG plot: {e:?}"))?;
    }
    Ok(svg)
}

fn render_conserved_visualization_svg(
    query_path: &Path,
    ref_path: &Path,
    query_mbp: f64,
    ref_mbp: f64,
    ani_label: &str,
    points: &[VisualizationPoint],
) -> Result<String> {
    // The conserved view uses native SVG cubic Bezier ribbons rather than
    // Plotters PathElement polylines. svg2pdf preserves these native paths in
    // the generated PDF, so the SVG and PDF have the same smooth geometry.
    const WIDTH: f64 = 1100.0;
    const HEIGHT: f64 = 520.0;
    const LEFT: f64 = 26.0;
    const RIGHT: f64 = 26.0;
    const QUERY_Y: f64 = 142.0;
    const REF_Y: f64 = 408.0;
    const TRACK_HEIGHT: f64 = 28.0;
    const MIN_RIBBON_PX: f64 = 0.55;

    let plot_width = WIDTH - LEFT - RIGHT;
    let max_mbp = query_mbp.max(ref_mbp).max(0.001);
    let x_scale = plot_width / max_mbp;
    let query_link_y = QUERY_Y + TRACK_HEIGHT * 0.5;
    let ref_link_y = REF_Y - TRACK_HEIGHT * 0.5;

    let mut svg = String::with_capacity(256 * 1024 + points.len() * 280);

    writeln!(
        svg,
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">"#
    )?;

    writeln!(svg, r#"<rect width="100%" height="100%" fill="white"/>"#)?;

    // Higher-identity ribbons are drawn first. Lower-identity ribbons are drawn
    // afterward, but their lower opacity keeps underlying strong alignments visible.
    let mut ordered_links = points.iter().collect::<Vec<_>>();
    ordered_links.sort_unstable_by(|a, b| cmp_f64(b.identity, a.identity));

    writeln!(svg, r#"<g id="mapping-ribbons">"#)?;

    for point in ordered_links {
        let mut q0 = LEFT + (point.query_start / 1_000_000.0) * x_scale;

        let mut q1 = LEFT + (point.query_end / 1_000_000.0) * x_scale;

        let mut r0 = LEFT + (point.ref_start / 1_000_000.0) * x_scale;

        let mut r1 = LEFT + (point.ref_end / 1_000_000.0) * x_scale;

        normalize_visible_interval(&mut q0, &mut q1, MIN_RIBBON_PX);

        normalize_visible_interval(&mut r0, &mut r1, MIN_RIBBON_PX);

        // Fixed vertical control levels keep every ribbon in the same curve
        // family and prevent inconsistent bends between neighboring mappings.
        let dy = ref_link_y - query_link_y;
        let control_offset = dy * 0.43;
        let c1y = query_link_y + control_offset;
        let c2y = ref_link_y - control_offset;

        let (red, green, blue, opacity) = identity_flame_style(point.identity);

        writeln!(
            svg,
            concat!(
                r#"<path d="M {q0:.4} {qy:.4} "#,
                r#"C {q0:.4} {c1y:.4}, {r0:.4} {c2y:.4}, {r0:.4} {ry:.4} "#,
                r#"L {r1:.4} {ry:.4} "#,
                r#"C {r1:.4} {c2y:.4}, {q1:.4} {c1y:.4}, {q1:.4} {qy:.4} Z" "#,
                r#"fill="rgb({red},{green},{blue})" "#,
                r#"fill-opacity="{opacity:.3}" stroke="none"/>"#
            ),
            q0 = q0,
            q1 = q1,
            r0 = r0,
            r1 = r1,
            qy = query_link_y,
            ry = ref_link_y,
            c1y = c1y,
            c2y = c2y,
            red = red,
            green = green,
            blue = blue,
            opacity = opacity,
        )?;
    }

    writeln!(svg, "</g>")?;

    // Neutral genome tracks expose unaligned regions as gray gaps.
    let query_width = query_mbp * x_scale;
    let ref_width = ref_mbp * x_scale;

    writeln!(
        svg,
        r##"<rect x="{LEFT:.3}" y="{:.3}" width="{query_width:.3}" height="{TRACK_HEIGHT:.3}" fill="#d6dbe0"/>"##,
        QUERY_Y - TRACK_HEIGHT * 0.5,
    )?;

    writeln!(
        svg,
        r##"<rect x="{LEFT:.3}" y="{:.3}" width="{ref_width:.3}" height="{TRACK_HEIGHT:.3}" fill="#d6dbe0"/>"##,
        REF_Y - TRACK_HEIGHT * 0.5,
    )?;

    // Aligned regions use the query-position color gradient. Each linked
    // reference block receives the same query-derived color.
    writeln!(svg, r#"<g id="aligned-blocks">"#)?;

    for point in points {
        let query_mid = 0.5 * (point.query_start + point.query_end);

        let RGBColor(red, green, blue) = genome_position_color(query_mid, query_mbp * 1_000_000.0);

        let mut q0 = LEFT + (point.query_start / 1_000_000.0) * x_scale;

        let mut q1 = LEFT + (point.query_end / 1_000_000.0) * x_scale;

        let mut r0 = LEFT + (point.ref_start / 1_000_000.0) * x_scale;

        let mut r1 = LEFT + (point.ref_end / 1_000_000.0) * x_scale;

        normalize_visible_interval(&mut q0, &mut q1, MIN_RIBBON_PX);

        normalize_visible_interval(&mut r0, &mut r1, MIN_RIBBON_PX);

        writeln!(
            svg,
            r#"<rect x="{q0:.4}" y="{:.3}" width="{:.4}" height="{TRACK_HEIGHT:.3}" fill="rgb({red},{green},{blue})"/>"#,
            QUERY_Y - TRACK_HEIGHT * 0.5,
            q1 - q0,
        )?;

        writeln!(
            svg,
            r#"<rect x="{r0:.4}" y="{:.3}" width="{:.4}" height="{TRACK_HEIGHT:.3}" fill="rgb({red},{green},{blue})"/>"#,
            REF_Y - TRACK_HEIGHT * 0.5,
            r1 - r0,
        )?;
    }

    writeln!(svg, "</g>")?;

    // Genome labels and ANI summary.
    writeln!(
        svg,
        r#"<text x="{LEFT:.1}" y="92" font-family="sans-serif" font-size="18" fill="black">{}</text>"#,
        xml_escape(&display_name(query_path)),
    )?;

    writeln!(
        svg,
        r#"<text x="{LEFT:.1}" y="478" font-family="sans-serif" font-size="18" fill="black">{}</text>"#,
        xml_escape(&display_name(ref_path)),
    )?;

    writeln!(
        svg,
        r#"<text x="550" y="48" text-anchor="middle" font-family="sans-serif" font-size="17" fill="black">{}</text>"#,
        xml_escape(ani_label),
    )?;

    append_native_identity_colorbar(&mut svg, 826.0, 63.0, 170.0, 12.0)?;
    append_native_scale_bar(&mut svg, LEFT, WIDTH - RIGHT, 492.0, x_scale, max_mbp)?;

    if points.is_empty() {
        writeln!(
            svg,
            r#"<text x="60" y="180" font-family="sans-serif" font-size="24" fill="black">No mapped fragments passed ANI thresholds</text>"#
        )?;
    }

    writeln!(svg, "</svg>")?;

    Ok(svg)
}

fn normalize_visible_interval(start: &mut f64, end: &mut f64, minimum_width: f64) {
    if *end < *start {
        std::mem::swap(start, end);
    }
    if *end - *start < minimum_width {
        let midpoint = 0.5 * (*start + *end);
        *start = midpoint - minimum_width * 0.5;
        *end = midpoint + minimum_width * 0.5;
    }
}

fn identity_flame_style(identity: f64) -> (u8, u8, u8, f64) {
    // Normalize the expected identity range of 80–100% to 0–1.
    let t = ((identity - 80.0) / 20.0).clamp(0.0, 1.0);

    // Nonlinear scaling increases visible separation between intermediate
    // and very high identities.
    let strength = t.powf(1.35);

    // Flame base color: #C16642.
    const FLAME_RED: f64 = 193.0;
    const FLAME_GREEN: f64 = 102.0;
    const FLAME_BLUE: f64 = 66.0;

    // Lower identities are blended substantially toward white.
    let pale_fraction = 0.78 * (1.0 - strength);

    let red = (FLAME_RED + (255.0 - FLAME_RED) * pale_fraction)
        .round()
        .clamp(0.0, 255.0) as u8;

    let green = (FLAME_GREEN + (255.0 - FLAME_GREEN) * pale_fraction)
        .round()
        .clamp(0.0, 255.0) as u8;

    let blue = (FLAME_BLUE + (255.0 - FLAME_BLUE) * pale_fraction)
        .round()
        .clamp(0.0, 255.0) as u8;

    // Opacity varies strongly with identity:
    // 80%  -> 0.16
    // 100% -> 0.92
    let opacity = 0.16 + 0.76 * strength;

    (red, green, blue, opacity)
}

fn xml_escape(text: &str) -> String {
    let mut escaped = String::with_capacity(text.len());
    for character in text.chars() {
        match character {
            '&' => escaped.push_str("&amp;"),
            '<' => escaped.push_str("&lt;"),
            '>' => escaped.push_str("&gt;"),
            '"' => escaped.push_str("&quot;"),
            '\'' => escaped.push_str("&apos;"),
            _ => escaped.push(character),
        }
    }
    escaped
}

fn append_native_identity_colorbar(
    svg: &mut String,
    x: f64,
    y: f64,
    width: f64,
    height: f64,
) -> Result<()> {
    writeln!(
        svg,
        concat!(
            r#"<defs>"#,
            r#"<linearGradient id="identity-gradient" "#,
            r#"x1="0%" x2="100%" y1="0%" y2="0%">"#,
            r#"<stop offset="0%" "#,
            r#"stop-color="rgb(241,218,208)" "#,
            r#"stop-opacity="0.16"/>"#,
            r#"<stop offset="50%" "#,
            r#"stop-color="rgb(218,160,137)" "#,
            r#"stop-opacity="0.46"/>"#,
            r#"<stop offset="100%" "#,
            r#"stop-color="rgb(193,102,66)" "#,
            r#"stop-opacity="0.92"/>"#,
            r#"</linearGradient>"#,
            r#"</defs>"#
        )
    )?;

    writeln!(
        svg,
        r#"<text x="{x:.1}" y="{:.1}" font-family="sans-serif" font-size="14" fill="black">Identity (%)</text>"#,
        y - 10.0,
    )?;

    writeln!(
        svg,
        r#"<rect x="{x:.1}" y="{y:.1}" width="{width:.1}" height="{height:.1}" fill="url(#identity-gradient)" stroke="black" stroke-width="1"/>"#
    )?;

    for (fraction, label) in [(0.0, "80"), (0.5, "90"), (1.0, "100")] {
        let tick_x = x + fraction * width;

        writeln!(
            svg,
            r#"<line x1="{tick_x:.1}" y1="{:.1}" x2="{tick_x:.1}" y2="{:.1}" stroke="black" stroke-width="1"/>"#,
            y + height,
            y + height + 5.0,
        )?;

        writeln!(
            svg,
            r#"<text x="{tick_x:.1}" y="{:.1}" text-anchor="middle" font-family="sans-serif" font-size="12" fill="black">{label}</text>"#,
            y + height + 22.0,
        )?;
    }

    Ok(())
}

fn append_native_scale_bar(
    svg: &mut String,
    plot_left: f64,
    plot_right: f64,
    y: f64,
    x_scale: f64,
    max_mbp: f64,
) -> Result<()> {
    let scale_mbp = nice_scale_bar_mbp(max_mbp);
    let bar_width = (scale_mbp * x_scale).min(plot_right - plot_left);
    if bar_width <= 0.0 {
        return Ok(());
    }

    let x1 = plot_right;
    let x0 = x1 - bar_width;
    let label = format_scale_bar_label(scale_mbp);

    writeln!(
        svg,
        r#"<g id="scale-bar" font-family="sans-serif" fill="black" stroke="black">"#
    )?;
    writeln!(
        svg,
        r#"<line x1="{x0:.1}" y1="{y:.1}" x2="{x1:.1}" y2="{y:.1}" stroke-width="1.5"/>"#
    )?;
    for x in [x0, x1] {
        writeln!(
            svg,
            r#"<line x1="{x:.1}" y1="{:.1}" x2="{x:.1}" y2="{:.1}" stroke-width="1.5"/>"#,
            y - 5.0,
            y + 5.0,
        )?;
    }
    writeln!(
        svg,
        r#"<text x="{:.1}" y="{:.1}" text-anchor="middle" font-size="13" stroke="none">{}</text>"#,
        0.5 * (x0 + x1),
        y + 20.0,
        xml_escape(&label),
    )?;
    writeln!(svg, "</g>")?;

    Ok(())
}

fn nice_scale_bar_mbp(max_mbp: f64) -> f64 {
    let target = (max_mbp * 0.22).max(0.001);
    let base = 10f64.powf(target.log10().floor());
    for multiplier in [5.0, 2.0, 1.0] {
        let candidate = multiplier * base;
        if candidate <= target {
            return candidate;
        }
    }
    base * 0.5
}

fn format_scale_bar_label(mbp: f64) -> String {
    if mbp >= 1.0 {
        format_measurement(mbp, "Mb")
    } else {
        format_measurement(mbp * 1_000.0, "kb")
    }
}

fn format_measurement(value: f64, unit: &str) -> String {
    let rounded = if value >= 100.0 {
        value.round()
    } else if value >= 10.0 {
        (value * 10.0).round() / 10.0
    } else {
        (value * 100.0).round() / 100.0
    };
    let mut label = format!("{rounded:.2}");
    while label.contains('.') && label.ends_with('0') {
        label.pop();
    }
    if label.ends_with('.') {
        label.pop();
    }
    format!("{label} {unit}")
}

fn draw_map_chart(
    area: &DrawingArea<SVGBackend<'_>, Shift>,
    _query_path: &Path,
    _ref_path: &Path,
    query_mbp: f64,
    ref_mbp: f64,
    ani_label: &str,
    points: &[VisualizationPoint],
) -> Result<()> {
    let mut map_chart = ChartBuilder::on(area)
        .caption(
            format!("Query-reference alignment positions ({ani_label})"),
            ("sans-serif", 16),
        )
        .margin(24)
        .x_label_area_size(62)
        .y_label_area_size(78)
        .build_cartesian_2d(0.0..query_mbp, 0.0..ref_mbp)
        .map_err(|e| anyhow::anyhow!("failed to build map chart: {e:?}"))?;

    map_chart
        .configure_mesh()
        .x_desc("Query position (Mb)")
        .y_desc("Reference position (Mb)")
        .axis_desc_style(("sans-serif", 24))
        .label_style(("sans-serif", 22))
        .draw()?;

    // Draw short alignment segments rather than only points,
    // giving a MUMmer-like map.
    map_chart
        .draw_series(points.iter().map(|point| {
            PathElement::new(
                vec![
                    (
                        point.query_start / 1_000_000.0,
                        point.ref_start / 1_000_000.0,
                    ),
                    (point.query_end / 1_000_000.0, point.ref_end / 1_000_000.0),
                ],
                identity_color(point.identity).stroke_width(2),
            )
        }))
        .map_err(|e| anyhow::anyhow!("failed to draw map segments: {e:?}"))?;

    if points.is_empty() {
        area.draw(&Text::new(
            "No mapped fragments passed ANI thresholds",
            (60, 80),
            ("sans-serif", 24).into_font(),
        ))
        .map_err(|e| anyhow::anyhow!("failed to draw empty-plot label: {e:?}"))?;
    }

    Ok(())
}

fn smooth_mapping_curve(
    point: &VisualizationPoint,
    query_link_y: f64,
    ref_link_y: f64,
    max_mbp: f64,
) -> Vec<(f64, f64)> {
    // Plotters' PathElement is a polyline rather than a native Bezier path. To
    // keep neighboring links visually coherent, every mapping must therefore use
    // exactly the same normalized curve family instead of independently selected
    // Bezier handles.
    //
    // Parameterize vertical position linearly and horizontal position with a
    // seventh-order smootherstep:
    //
    //   s(t) = 35t^4 - 84t^5 + 70t^6 - 20t^7
    //
    // The first three derivatives vanish at both endpoints. Links therefore leave
    // and enter the genome tracks vertically, with zero endpoint curvature and no
    // visible shoulder. Because the same s(t) is used for every mapping, adjacent
    // colinear mappings remain parallel instead of opening artificial white wedges.
    const CURVE_STEPS: usize = 128;

    let scale = max_mbp.max(f64::EPSILON);
    let x0 = (0.5 * (point.query_start + point.query_end) / 1_000_000.0) / scale;
    let x1 = (0.5 * (point.ref_start + point.ref_end) / 1_000_000.0) / scale;
    let y0 = query_link_y;
    let y1 = ref_link_y;

    (0..=CURVE_STEPS)
        .map(|step| {
            let t = step as f64 / CURVE_STEPS as f64;
            let t2 = t * t;
            let t3 = t2 * t;
            let t4 = t2 * t2;
            let t5 = t4 * t;
            let t6 = t3 * t3;
            let t7 = t6 * t;

            let s = 35.0 * t4 - 84.0 * t5 + 70.0 * t6 - 20.0 * t7;
            let x = x0 + (x1 - x0) * s;
            let y = y0 + (y1 - y0) * t;

            (x * scale, y)
        })
        .collect()
}

fn draw_homology_ribbon_chart(
    area: &DrawingArea<SVGBackend<'_>, Shift>,
    query_path: &Path,
    ref_path: &Path,
    query_mbp: f64,
    ref_mbp: f64,
    ani_label: &str,
    points: &[VisualizationPoint],
) -> Result<()> {
    let max_mbp = query_mbp.max(ref_mbp).max(0.001);
    let query_y = 0.78;
    let ref_y = 0.22;
    let bar_half = 0.030;

    // No configure_mesh(): this conserved-region view intentionally has no axes,
    // ticks, plot frame, or grid. The MUMmer-like map remains a separate plot.
    let mut chart = ChartBuilder::on(area)
        .margin(20)
        .build_cartesian_2d(0.0..max_mbp, 0.0..1.0)
        .map_err(|e| anyhow::anyhow!("failed to build homology ribbon chart: {e:?}"))?;

    // Draw high-identity links first and lower-identity links last. The weaker,
    // paler mappings are therefore not hidden beneath the stronger mappings, while
    // the strong mappings remain visible through the links' transparency.
    let mut ordered_links = points.iter().collect::<Vec<_>>();
    ordered_links.sort_unstable_by(|a, b| cmp_f64(b.identity, a.identity));

    // Identity is encoded by link shade. Use a thin one-pixel centerline so dense
    // comparisons remain readable and individual rearrangements do not merge into
    // broad gray bands.
    chart
        .draw_series(ordered_links.into_iter().map(|point| {
            PathElement::new(
                smooth_mapping_curve(point, query_y - bar_half, ref_y + bar_half, max_mbp),
                ShapeStyle::from(&identity_link_color(point.identity).mix(0.60)).stroke_width(1),
            )
        }))
        .map_err(|e| anyhow::anyhow!("failed to draw homology links: {e:?}"))?;

    // Neutral backgrounds make unaligned sequence immediately visible.
    let unaligned_track = RGBColor(214, 219, 224);
    chart
        .draw_series([
            Rectangle::new(
                [(0.0, query_y - bar_half), (query_mbp, query_y + bar_half)],
                ShapeStyle::from(&unaligned_track).filled().stroke_width(1),
            ),
            Rectangle::new(
                [(0.0, ref_y - bar_half), (ref_mbp, ref_y + bar_half)],
                ShapeStyle::from(&unaligned_track).filled().stroke_width(1),
            ),
        ])
        .map_err(|e| anyhow::anyhow!("failed to draw genome tracks: {e:?}"))?;

    // pyGenomeViz-style positional coloring: color is determined by the query
    // coordinate, not by mapping index. Nearby query regions therefore form a
    // smooth viridis-like gradient. The linked reference interval receives the
    // exact same color, which makes translocations and cross-mapping visible.
    // Neutral track regions remain visible wherever no alignment is present.
    chart
        .draw_series(points.iter().map(|point| {
            let query_mid = 0.5 * (point.query_start + point.query_end);
            let color = genome_position_color(query_mid, query_mbp * 1_000_000.0);
            Rectangle::new(
                [
                    (point.query_start / 1_000_000.0, query_y - bar_half),
                    (point.query_end / 1_000_000.0, query_y + bar_half),
                ],
                color.filled(),
            )
        }))
        .map_err(|e| anyhow::anyhow!("failed to color query aligned blocks: {e:?}"))?;

    chart
        .draw_series(points.iter().map(|point| {
            let query_mid = 0.5 * (point.query_start + point.query_end);
            let color = genome_position_color(query_mid, query_mbp * 1_000_000.0);
            Rectangle::new(
                [
                    (point.ref_start / 1_000_000.0, ref_y - bar_half),
                    (point.ref_end / 1_000_000.0, ref_y + bar_half),
                ],
                color.filled(),
            )
        }))
        .map_err(|e| anyhow::anyhow!("failed to color reference aligned blocks: {e:?}"))?;

    chart
        .draw_series([
            Text::new(
                display_name(query_path),
                (0.0, query_y + 0.095),
                ("sans-serif", 18).into_font(),
            ),
            Text::new(
                display_name(ref_path),
                (0.0, ref_y - 0.115),
                ("sans-serif", 18).into_font(),
            ),
            Text::new(
                ani_label.to_owned(),
                (max_mbp * 0.5, 0.96),
                ("sans-serif", 17).into_font(),
            ),
        ])
        .map_err(|e| anyhow::anyhow!("failed to draw homology map labels: {e:?}"))?;

    // This is the requested line-identity heat indicator, not a separate heatmap.
    draw_identity_colorbar(area)?;

    if points.is_empty() {
        area.draw(&Text::new(
            "No mapped fragments passed ANI thresholds",
            (60, 80),
            ("sans-serif", 24).into_font(),
        ))
        .map_err(|e| anyhow::anyhow!("failed to draw empty homology-map label: {e:?}"))?;
    }

    Ok(())
}

fn draw_identity_chart(
    area: &DrawingArea<SVGBackend<'_>, Shift>,
    query_mbp: f64,
    results: &[AniResult],
    points: &[VisualizationPoint],
) -> Result<()> {
    let y_min = points
        .iter()
        .map(|point| point.identity)
        .fold(100.0f64, f64::min)
        .floor()
        .min(99.0)
        .max(75.0);
    let y_max = 100.1;
    let mut identity_chart = ChartBuilder::on(area)
        .caption("Fragment identity by query position", ("sans-serif", 16))
        .margin(24)
        .x_label_area_size(62)
        .y_label_area_size(78)
        .build_cartesian_2d(0.0..query_mbp, y_min..y_max)
        .map_err(|e| anyhow::anyhow!("failed to build identity chart: {e:?}"))?;
    identity_chart
        .configure_mesh()
        .x_desc("Query position (Mb)")
        .y_desc("Identity")
        .axis_desc_style(("sans-serif", 24))
        .label_style(("sans-serif", 22))
        .draw()?;
    identity_chart
        .draw_series(points.iter().map(|point| {
            Circle::new(
                (point.query_mid() / 1_000_000.0, point.identity),
                2,
                identity_color(point.identity).filled(),
            )
        }))
        .map_err(|e| anyhow::anyhow!("failed to draw identity points: {e:?}"))?;

    if let Some(result) = results.first() {
        identity_chart
            .draw_series(LineSeries::new(
                vec![(0.0, result.ani), (query_mbp, result.ani)],
                &RED,
            ))
            .map_err(|e| anyhow::anyhow!("failed to draw ANI line: {e:?}"))?;
    }
    Ok(())
}

fn draw_conserved_chart(
    area: &DrawingArea<SVGBackend<'_>, Shift>,
    query_path: &Path,
    ref_path: &Path,
    query_mbp: f64,
    ref_mbp: f64,
    ani_label: &str,
    points: &[VisualizationPoint],
) -> Result<()> {
    draw_homology_ribbon_chart(
        area, query_path, ref_path, query_mbp, ref_mbp, ani_label, points,
    )
}

fn draw_identity_colorbar(area: &DrawingArea<SVGBackend<'_>, Shift>) -> Result<()> {
    let (area_width, _) = area.dim_in_pixel();
    let bar_width = 170i32;
    let bar_height = 12i32;
    let x0 = (area_width as i32 - 265).max(140);
    let y0 = 64i32;

    area.draw(&Text::new(
        "Identity (%)",
        (x0, y0 - 12),
        ("sans-serif", 14).into_font(),
    ))
    .map_err(|e| anyhow::anyhow!("failed to draw identity colorbar title: {e:?}"))?;

    for pixel in 0..bar_width {
        let identity = 80.0 + 20.0 * pixel as f64 / (bar_width - 1) as f64;
        area.draw(&Rectangle::new(
            [(x0 + pixel, y0), (x0 + pixel + 1, y0 + bar_height)],
            identity_link_color(identity).filled(),
        ))
        .map_err(|e| anyhow::anyhow!("failed to draw identity colorbar gradient: {e:?}"))?;
    }

    area.draw(&Rectangle::new(
        [(x0, y0), (x0 + bar_width, y0 + bar_height)],
        BLACK.stroke_width(1),
    ))
    .map_err(|e| anyhow::anyhow!("failed to draw identity colorbar frame: {e:?}"))?;

    for (fraction, label) in [(0.0, "80"), (0.5, "90"), (1.0, "100")] {
        let x = x0 + (fraction * bar_width as f64).round() as i32;
        area.draw(&PathElement::new(
            vec![(x, y0 + bar_height), (x, y0 + bar_height + 5)],
            BLACK,
        ))
        .map_err(|e| anyhow::anyhow!("failed to draw identity colorbar tick: {e:?}"))?;
        area.draw(&Text::new(
            label,
            (x - 8, y0 + bar_height + 22),
            ("sans-serif", 12).into_font(),
        ))
        .map_err(|e| anyhow::anyhow!("failed to draw identity colorbar label: {e:?}"))?;
    }
    Ok(())
}

fn ani_label(results: &[AniResult]) -> String {
    results
        .first()
        .map(|result| {
            format!(
                "ANI {:.4}%, mapped {}/{}",
                result.ani, result.mapped_fragments, result.total_query_fragments
            )
        })
        .unwrap_or_else(|| "no ANI result passed thresholds".to_string())
}

fn identity_color(identity: f64) -> HSLColor {
    let t = ((identity - 80.0) / 20.0).clamp(0.0, 1.0);
    HSLColor((220.0 - 180.0 * t) / 360.0, 0.75, 0.45)
}

fn identity_link_color(identity: f64) -> HSLColor {
    // Wide grayscale range: very pale at 80%, nearly black at 100%. The links are
    // still translucent, but this larger luminance span keeps identity differences
    // visible after alpha blending and in regions with overlapping mappings.
    let t = ((identity - 80.0) / 20.0).clamp(0.0, 1.0);
    HSLColor(0.0, 0.0, 0.92 - 0.86 * t)
}

fn genome_position_color(position: f64, genome_len: f64) -> RGBColor {
    // Viridis-like positional gradient used for aligned genome blocks.
    // The same query-derived color is applied to both ends of an alignment.
    // Stops approximate matplotlib/pyGenomeViz viridis without adding a dependency.
    const STOPS: &[(f64, (u8, u8, u8))] = &[
        (0.00, (68, 1, 84)),
        (0.13, (71, 44, 122)),
        (0.25, (59, 82, 139)),
        (0.38, (44, 113, 142)),
        (0.50, (33, 145, 140)),
        (0.63, (39, 173, 129)),
        (0.75, (92, 200, 99)),
        (0.88, (170, 220, 50)),
        (1.00, (253, 231, 37)),
    ];

    let t = if genome_len > 0.0 {
        (position / genome_len).clamp(0.0, 1.0)
    } else {
        0.0
    };

    let mut right = 1usize;
    while right < STOPS.len() && t > STOPS[right].0 {
        right += 1;
    }
    if right >= STOPS.len() {
        let (r, g, b) = STOPS[STOPS.len() - 1].1;
        return RGBColor(r, g, b);
    }

    let left = right - 1;
    let (t0, (r0, g0, b0)) = STOPS[left];
    let (t1, (r1, g1, b1)) = STOPS[right];
    let u = if t1 > t0 { (t - t0) / (t1 - t0) } else { 0.0 };

    let lerp = |a: u8, b: u8| -> u8 { (a as f64 + (b as f64 - a as f64) * u).round() as u8 };

    RGBColor(lerp(r0, r1), lerp(g0, g1), lerp(b0, b1))
}

fn display_name(path: &Path) -> String {
    path.file_name()
        .and_then(|name| name.to_str())
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| path.display().to_string())
}
