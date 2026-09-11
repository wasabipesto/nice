//! Partition audit for the `fields` and `chunks` tables.
//!
//! Every progress figure the API serves assumes that the fields of a base
//! partition its range `[range_start, range_end)`: no gaps, no overlaps,
//! first start at the base start, last end at the base end, and
//! `range_size = range_end - range_start` on every row. The leaderboard and
//! rate caches sum `range_size` per submission, `get_count_checked_by_range`
//! sums it over a window, and the per-base totals sum the chunk rows - all
//! of which double-count on an overlap and under-count on a gap. Chunks are
//! assumed to partition the base the same way, and each field is assumed
//! to lie inside its chunk: `reassign_fields_to_chunks` matches a field to
//! the chunk containing its *start point only*, which is correct exactly
//! when no field straddles a chunk boundary.
//!
//! The generators (`break_range_into_fields`, `group_fields_into_chunks`)
//! are tested in isolation, but nothing re-checked the rows actually in the
//! database until this module. The queries are one sorted window scan per
//! table per base, so they run as part of `nice_jobs --full` (and on demand
//! with `nice_jobs --audit`) rather than on every incremental run.

use super::*;
use diesel::sql_query;
use diesel::sql_types::{BigInt, Integer, Nullable, Numeric};

/// The result of scanning one table's rows for one base in range order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PartitionAudit {
    /// Rows for the base.
    pub rows: u64,
    /// Rows whose start is past the largest end seen so far (a hole).
    pub gaps: u64,
    /// Rows whose start is before the largest end seen so far (double cover).
    pub overlaps: u64,
    /// Rows with `range_size != range_end - range_start` or an empty range.
    pub size_mismatches: u64,
    /// Smallest `range_start`, if any rows.
    pub first_start: Option<u128>,
    /// Largest `range_end`, if any rows.
    pub last_end: Option<u128>,
    /// `SUM(range_size)`.
    pub total_size: u128,
}

impl PartitionAudit {
    /// Human-readable problems relative to the base range `[start, end)`.
    /// Empty when the rows partition it exactly.
    #[must_use]
    pub fn problems(&self, start: u128, end: u128) -> Vec<String> {
        let mut out = Vec::new();
        if self.rows == 0 {
            out.push("no rows".to_string());
            return out;
        }
        if self.gaps > 0 {
            out.push(format!("{} gap(s)", self.gaps));
        }
        if self.overlaps > 0 {
            out.push(format!("{} overlap(s)", self.overlaps));
        }
        if self.size_mismatches > 0 {
            out.push(format!(
                "{} row(s) with range_size != end - start",
                self.size_mismatches
            ));
        }
        if self.first_start != Some(start) {
            out.push(format!(
                "first start {:?} != base start {start}",
                self.first_start
            ));
        }
        if self.last_end != Some(end) {
            out.push(format!("last end {:?} != base end {end}", self.last_end));
        }
        if self.total_size != end - start {
            out.push(format!(
                "total size {} != base size {}",
                self.total_size,
                end - start
            ));
        }
        out
    }
}

/// Fields that are not properly inside a chunk of their own base.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct FieldChunkAudit {
    /// Fields with `chunk_id IS NULL`.
    pub unassigned: u64,
    /// Fields whose range is not contained in their chunk's range.
    pub straddling: u64,
    /// Fields whose chunk belongs to another base.
    pub wrong_base: u64,
}

impl FieldChunkAudit {
    #[must_use]
    pub fn problems(&self) -> Vec<String> {
        let mut out = Vec::new();
        if self.unassigned > 0 {
            out.push(format!("{} field(s) with no chunk", self.unassigned));
        }
        if self.straddling > 0 {
            out.push(format!(
                "{} field(s) not inside their chunk",
                self.straddling
            ));
        }
        if self.wrong_base > 0 {
            out.push(format!(
                "{} field(s) whose chunk is in another base",
                self.wrong_base
            ));
        }
        out
    }
}

/// Everything the audit checks for one base, with the problems already
/// rendered so the jobs binary only has to print them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BaseAudit {
    pub base: u32,
    pub fields: PartitionAudit,
    pub chunks: PartitionAudit,
    pub fields_in_chunks: FieldChunkAudit,
    pub problems: Vec<String>,
}

#[derive(QueryableByName)]
struct PartitionRow {
    #[diesel(sql_type = BigInt)]
    rows: i64,
    #[diesel(sql_type = BigInt)]
    gaps: i64,
    #[diesel(sql_type = BigInt)]
    overlaps: i64,
    #[diesel(sql_type = BigInt)]
    size_mismatches: i64,
    #[diesel(sql_type = Nullable<Numeric>)]
    first_start: Option<BigDecimal>,
    #[diesel(sql_type = Nullable<Numeric>)]
    last_end: Option<BigDecimal>,
    #[diesel(sql_type = Numeric)]
    total_size: BigDecimal,
}

fn i64_to_u64(i: i64) -> Result<u64> {
    u64::try_from(i).map_err(|e| anyhow!("{e}"))
}

/// Scan `table` (`"fields"` or `"chunks"`) for one base in range order.
///
/// `prev_end` is the running maximum of every earlier row's end, not just
/// the previous row's, so a row nested entirely inside an earlier one still
/// counts as an overlap.
fn audit_table(conn: &mut PgConnection, table: &str, base: u32) -> Result<PartitionAudit> {
    assert!(
        table == "fields" || table == "chunks",
        "audit_table takes a fixed table name"
    );
    let base = conversions::u32_to_i32(base)?;
    let row: PartitionRow = sql_query(format!(
        "WITH ordered AS (
             SELECT range_start, range_end, range_size,
                    MAX(range_end) OVER (
                        ORDER BY range_start, range_end
                        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                    ) AS prev_end
             FROM {table} WHERE base_id = $1
         )
         SELECT COUNT(*)::BIGINT AS rows,
                COUNT(*) FILTER (WHERE prev_end IS NOT NULL AND range_start > prev_end)::BIGINT AS gaps,
                COUNT(*) FILTER (WHERE prev_end IS NOT NULL AND range_start < prev_end)::BIGINT AS overlaps,
                COUNT(*) FILTER (WHERE range_end <= range_start
                                    OR range_size <> range_end - range_start)::BIGINT AS size_mismatches,
                MIN(range_start) AS first_start,
                MAX(range_end) AS last_end,
                COALESCE(SUM(range_size), 0) AS total_size
         FROM ordered"
    ))
    .bind::<Integer, _>(base)
    .get_result(conn)
    .map_err(|e| anyhow!("{e}"))?;

    Ok(PartitionAudit {
        rows: i64_to_u64(row.rows)?,
        gaps: i64_to_u64(row.gaps)?,
        overlaps: i64_to_u64(row.overlaps)?,
        size_mismatches: i64_to_u64(row.size_mismatches)?,
        first_start: row
            .first_start
            .map(conversions::bigdec_to_u128)
            .transpose()?,
        last_end: row.last_end.map(conversions::bigdec_to_u128).transpose()?,
        total_size: conversions::bigdec_to_u128(row.total_size)?,
    })
}

/// Do the fields of `base` partition a range? (Compare against the base
/// record with [`PartitionAudit::problems`].)
pub fn audit_fields_partition(conn: &mut PgConnection, base: u32) -> Result<PartitionAudit> {
    audit_table(conn, "fields", base)
}

/// Do the chunks of `base` partition a range?
pub fn audit_chunks_partition(conn: &mut PgConnection, base: u32) -> Result<PartitionAudit> {
    audit_table(conn, "chunks", base)
}

#[derive(QueryableByName)]
struct FieldChunkRow {
    #[diesel(sql_type = BigInt)]
    unassigned: i64,
    #[diesel(sql_type = BigInt)]
    straddling: i64,
    #[diesel(sql_type = BigInt)]
    wrong_base: i64,
}

/// Is every field of `base` inside a chunk of the same base?
pub fn audit_fields_within_chunks(conn: &mut PgConnection, base: u32) -> Result<FieldChunkAudit> {
    let base = conversions::u32_to_i32(base)?;
    let row: FieldChunkRow = sql_query(
        "SELECT COUNT(*) FILTER (WHERE f.chunk_id IS NULL)::BIGINT AS unassigned,
                COUNT(*) FILTER (WHERE c.id IS NOT NULL
                                    AND (f.range_start < c.range_start
                                         OR f.range_end > c.range_end))::BIGINT AS straddling,
                COUNT(*) FILTER (WHERE c.id IS NOT NULL AND c.base_id <> f.base_id)::BIGINT AS wrong_base
         FROM fields f LEFT JOIN chunks c ON c.id = f.chunk_id
         WHERE f.base_id = $1",
    )
    .bind::<Integer, _>(base)
    .get_result(conn)
    .map_err(|e| anyhow!("{e}"))?;
    Ok(FieldChunkAudit {
        unassigned: i64_to_u64(row.unassigned)?,
        straddling: i64_to_u64(row.straddling)?,
        wrong_base: i64_to_u64(row.wrong_base)?,
    })
}

/// Run all three checks for one base against its base record.
pub fn audit_base(conn: &mut PgConnection, base: &BaseRecord) -> Result<BaseAudit> {
    let fields = audit_fields_partition(conn, base.base)?;
    let chunks = audit_chunks_partition(conn, base.base)?;
    let fields_in_chunks = audit_fields_within_chunks(conn, base.base)?;
    let mut problems = Vec::new();
    problems.extend(
        fields
            .problems(base.range_start, base.range_end)
            .into_iter()
            .map(|p| format!("fields: {p}")),
    );
    problems.extend(
        chunks
            .problems(base.range_start, base.range_end)
            .into_iter()
            .map(|p| format!("chunks: {p}")),
    );
    problems.extend(
        fields_in_chunks
            .problems()
            .into_iter()
            .map(|p| format!("fields/chunks: {p}")),
    );
    Ok(BaseAudit {
        base: base.base,
        fields,
        chunks,
        fields_in_chunks,
        problems,
    })
}
