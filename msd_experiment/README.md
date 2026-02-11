# MSD Filter Effectiveness Experiment

This tool computes the effectiveness of the MSD (Most Significant Digit) filter across different bases, with SQLite caching support for resumability, progressive refinement, and parallel processing.

## Features

- **Persistent Caching**: All computation results are cached in SQLite, enabling crash recovery
- **Progressive Refinement**: Increase depth without redoing previous work
- **Parallel Processing**: Compute multiple bases in parallel using Rayon
- **Exportable Results**: Export cached data to JSON for visualization or analysis
- **Flexible Parameters**: Adjust max depth, min range size, and subdivision factor

## Usage

### Basic Computation

Compute the MSD filter effectiveness for a single base:

```bash
cargo run -r -p nice_msd_experiment -- --base 40 --max-depth 20
```

### Compute Multiple Bases

Compute a range of bases sequentially:

```bash
cargo run -r -p nice_msd_experiment -- --base 10 --base-end 50 --max-depth 20
```

### Parallel Processing

Compute multiple bases in parallel for better performance:

```bash
cargo run -r -p nice_msd_experiment -- --base 10 --base-end 100 --max-depth 20 --parallel
```

### Progressive Refinement

Start with a quick, shallow computation:

```bash
cargo run -r -p nice_msd_experiment -- --base 40 --max-depth 10
```

Then increase depth to refine results (reuses previous work):

```bash
cargo run -r -p nice_msd_experiment -- --base 40 --max-depth 30
```

And finally run overnight with ideal depth:

```bash
cargo run -r -p nice_msd_experiment -- --base 40 --max-depth 50
```

### View Statistics

Show cached data statistics for a base:

```bash
cargo run -r -p nice_msd_experiment -- --stats --base 40
```

Or for a range:

```bash
cargo run -r -p nice_msd_experiment -- --stats --base 10 --base-end 50
```

### Export Results

Export cached data for a base to JSON:

```bash
cargo run -r -p nice_msd_experiment -- --export 40
```

This creates `msd_cache_base_40.json` with all cached subranges.

### Clear Cache

Clear cached data for a specific base:

```bash
cargo run -r -p nice_msd_experiment -- --clear-cache 40
```

## Command-Line Options

```
Options:
  -b, --base <BASE>
          Base to compute (or start of range if --base-end is specified) [default: 10]
      --base-end <BASE_END>
          End of base range (exclusive). If specified, computes all bases from --base to this value
  -m, --max-depth <MAX_DEPTH>
          Maximum recursion depth [default: 20]
  -r, --min-range-size <MIN_RANGE_SIZE>
          Minimum range size before stopping recursion [default: 10000]
  -s, --subdivision-factor <SUBDIVISION_FACTOR>
          Subdivision factor (how many parts to split each range into) [default: 2]
  -d, --db-path <DB_PATH>
          Path to SQLite database file [default: msd_cache.db]
  -p, --parallel
          Use parallel processing for multiple bases
      --stats
          Show statistics for cached data instead of computing
      --export <EXPORT>
          Export cached data for a base to JSON
      --clear-cache <CLEAR_CACHE>
          Clear cache for a specific base
  -v, --verbose
          Verbose output
  -h, --help
          Print help
  -V, --version
          Print version
```

## How It Works

### Caching Strategy

The tool uses SQLite to cache intermediate computation results:

1. **Cache Key**: Each subrange is identified by `(base, range_start, range_end)`
2. **Depth Tracking**: Stores the maximum depth computed for each range
3. **Reusability**: If cached depth ≥ requested depth, uses cached result
4. **Selective Caching**: Only caches terminal nodes and periodic checkpoints to reduce overhead

### Database Schema

```sql
CREATE TABLE msd_cache (
    base INTEGER NOT NULL,
    range_start TEXT NOT NULL,      -- u128 as string
    range_end TEXT NOT NULL,        -- u128 as string
    max_depth INTEGER NOT NULL,     -- how deep this was computed
    min_range_size TEXT NOT NULL,
    subdivision_factor INTEGER NOT NULL,
    valid_size TEXT NOT NULL,       -- u128 result as string
    computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (base, range_start, range_end)
);
```

### Performance Tips

1. **Start Shallow**: Begin with `--max-depth 10` to get quick results
2. **Use Parallel**: For multiple bases, always use `--parallel`
3. **Incremental Depth**: Gradually increase depth (10 → 20 → 30 → 50)
4. **Database Location**: For best I/O, keep `msd_cache.db` on an SSD

### Technical Notes

**Concurrent Access Handling**

The tool uses several techniques to handle concurrent database writes safely:

- **WAL Mode**: SQLite Write-Ahead Logging for better concurrent access
- **Connection Pool**: Up to 32 concurrent connections with r2d2
- **Busy Timeout**: 5-second timeout for lock conflicts
- **Retry Logic**: Automatic retry with exponential backoff (up to 10 attempts)

This allows parallel processing of multiple bases without database lock errors. Each thread gets its own connection from the pool, and writes are automatically retried if conflicts occur.

**Caching Strategy**

To minimize database overhead, the tool selectively caches:
- Terminal nodes (depth limit or size limit reached)
- Filtered ranges (where MSD filter eliminates entire range)
- Periodic checkpoints (every 5 depth levels)
- Top-level results (depth 0)

This reduces write operations by ~90% while maintaining crash recovery capability.

## Integration with Other Scripts

The cached data can be imported into other analysis tools. The JSON export includes:

```json
[
  {
    "base": 40,
    "range_start": 1916284264916,
    "range_end": 6553600000000,
    "max_depth": 20,
    "min_range_size": 10000,
    "subdivision_factor": 2,
    "valid_size": 4637315735084
  }
]
```

This can be used to compare MSD filter effectiveness with LSD and residue filters.

## Examples

### Quick Survey

Get a rough estimate across all bases quickly:

```bash
cargo run -r -p nice_msd_experiment -- --base 10 --base-end 100 --max-depth 10 --parallel
```

### Deep Dive on One Base

Thoroughly analyze a single base:

```bash
cargo run -r -p nice_msd_experiment -- --base 40 --max-depth 50 --verbose
```

### Overnight Run

Compute all bases at high depth overnight:

```bash
cargo run -r -p nice_msd_experiment -- --base 10 --base-end 100 --max-depth 40 --parallel
```

If it crashes, just run the same command again - it will resume from where it left off!