//! Makes the build commit available to the binary at compile time.
//!
//! The commit is read from `NICE_BUILD_COMMIT` rather than from git: the
//! Docker build context excludes `.git` (see `.dockerignore`), so a
//! `git rev-parse` here would find nothing in the image build. The image
//! passes the value in as a build argument instead. Builds that set nothing
//! (a plain `cargo build`) simply report no commit.
fn main() {
    // Without this, cargo would not rebuild when only the commit changes,
    // and the binary would keep reporting whatever was cached.
    println!("cargo:rerun-if-env-changed=NICE_BUILD_COMMIT");
}
