# TurboANI

TurboANI is a super fast ANI estimation algorithm implemented in pure Rust:

- `simd-minimizers` for canonical minimizer positions and super-k-mer window coordinates. Note that the super-k-mer windows and corresponding minimizer positions were retained. 
- `tab-hash::Tab64Twisted` for deterministic 64-bit tabulation-hashed minimizer. This step is a rehash to obtain true pseudo-randomness via twisted tabulation hashing. 
- A MUMmer-style diagonal clustering followed by minimap2-style fast chaining for L1 candidate window screening.
- A cache-friendly exact L2 bottom-sketch slide mapper with local coordinate indices and a two-level bitset pivot.
- `plotters` plus `svg2pdf` for single-pair PDF visualizations.

The main algorithm flow:

1. Build a reference minimizer lookup.
2. L1: cluster query-fragment minimizer seed hits into candidate reference intervals (via diagnoal clustering followed by minimap2 chaining or the much slower optimal ChainX co-linear chaining).
3. L2: slide query-length super-windows over each L1 interval and score each placement.
4. Keep best-hit and reference-bin reciprocal filters before averaging ANI.
5. The split mode allows query mapping to a small subset of references to reduce RAM. 

L2 scoring uses a new mashmap-like incremental bottom-sketch slide mapper: a query fragment's unique minimizer seed the bottom-k union, and each candidate reference super-window is updated as the window slides. The active implementation uses local coordinate compression plus a summary bitset to move the bottom-k pivot with word-level operations instead of a tree lookup on every insert/delete.

The final Mash distance and confidence-bound calculation is cached exactly by `(sketch_size, best_shared)`, so each L2 candidate performs a table lookup rather than recomputing the same binomial-bound math.

`simd-minimizers` uses ntHash internally with SIMD to choose minimizer positions. The turboani binary takes the returned canonical minimizer k-mer value and applies `Tab64Twisted` tabulation hashing once. That minimizer is reused for L1 lookup and exact L2 bottom-sketch comparison. The tabulation table is deterministic by default and controlled by `--tabSeed`.

## Build and run

```bash
git clone https://github.com/jianshu93/TurboANI
RUSTFLAGS="-C target-cpu=native" cargo build --release
./target/release/turboani -h
```

### single pair comparsion
Single-pair visualization writes a PDF with a query/reference map and fragment identity panel:

```bash
./target/release/turboani \
  -q query.fa \
  -r reference.fa \
  -o pair.tsv \
  --visualize pair.pdf
```
`--visualize` intentionally supports only one `-q` and one `-r`; it rejects list mode.


This it the plot.

<div align="center">
  <img width="100%" src ="Figures.jpg">
</div>


### Many comparsions
List mode is also supported (one file per line, .gz supported):

```bash
./target/release/turboani \
  --ql queries.txt \
  --rl references.txt \
  -o turboani.tsv \
```

Output columns match FastANI:

```text
query_path  reference_path  ani  mapped_fragments  total_query_fragments
```

## Separate fastANI binary
FastANI compatibility mode is a separate binary that implements the original FastANI algorithm:

```bash
./target/release/fastani -- \
  --ql queries.txt \
  --rl references.txt \
  -o fastani-style.tsv \
```


## Useful knobs:

- `--fragLen 3000`
- `-k 16`
- `--minIdentity 80`
- `--minFraction 0.2`
- `--tabSeed 42`
- `--windowSize N` to override p-value-derived minimizer window.
- `--ignoreTopPercent P` to ignore the most frequent minimizers during L1 lookup.
- `--visualize pair.pdf` to write a single-pair PDF map/identity plot.

Timing is collected internally and emitted only in debug mode. For a concise timing/counter summary, run with `RUST_LOG=debug`.

Rayon is used at three levels where the data are independent: reference genome indexing across reference files, query-file comparisons, and fragment mapping inside each query file.

## Notes

The implementation skips minimizer windows spanning ambiguous bases before passing sequence runs to `simd-minimizers`. This is usually preferable for Rust SIMD packing and keeps `N`-rich phage assemblies from creating artificial seeds.

## References
Jain, C., Rodriguez-R, L.M., Phillippy, A.M., Konstantinidis, K.T. and Aluru, S., 2018. High throughput ANI analysis of 90K prokaryotic genomes reveals clear species boundaries. Nature communications, 9(1), p.5114.

Jain, C., Dilthey, A., Koren, S., Aluru, S. and Phillippy, A.M., 2018. A fast approximate algorithm for mapping long reads to large reference databases. Journal of Computational Biology, 25(7), pp.766-779.

Delcher, A.L., Phillippy, A., Carlton, J. and Salzberg, S.L., 2002. Fast algorithms for large-scale genome alignment and comparison. Nucleic acids research, 30(11), pp.2478-2483.

Li, H., 2016. Minimap and miniasm: fast mapping and de novo assembly for noisy long sequences. Bioinformatics, 32(14), pp.2103-2110.

Marchini, S. and Vigna, S., 2020. Compact Fenwick trees for dynamic ranking and selection. Software: Practice and Experience, 50(7), pp.1184-1202.
