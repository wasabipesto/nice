#![allow(dead_code)]

use super::*;

table! {
    benchmarks (id) {
        id -> BigInt,
        submit_time -> Timestamptz,
        username -> Varchar,
        user_ip -> Varchar,
        client_version -> Varchar,
        data -> Jsonb,
    }
}

#[derive(Insertable)]
#[diesel(table_name = benchmarks)]
struct BenchmarkNew {
    username: String,
    user_ip: String,
    client_version: String,
    data: Value,
}

/// Store one uploaded benchmark report. `client_version` is extracted from
/// the report by the caller (it lives inside `data` too; the column exists
/// for cheap filtering).
pub fn insert_benchmark(
    conn: &mut PgConnection,
    input_username: String,
    input_user_ip: String,
    input_client_version: String,
    input_data: Value,
) -> Result<i64> {
    use self::benchmarks::dsl::*;

    let row = BenchmarkNew {
        username: input_username,
        user_ip: input_user_ip,
        client_version: input_client_version,
        data: input_data,
    };
    let inserted_id: i64 = diesel::insert_into(benchmarks)
        .values(&row)
        .returning(id)
        .get_result(conn)
        .map_err(|e| anyhow!("{e}"))?;
    Ok(inserted_id)
}
