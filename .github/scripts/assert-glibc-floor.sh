#!/usr/bin/env bash
# Assert that an ELF binary needs no glibc symbol version above a floor.
#
# The floor is a property of the build environment, not of the source, so it
# is checked on the artifact: every GLIBC_x.y version the binary requires
# must be <= the floor. readelf reads any ELF architecture, unlike the host
# objdump. Usage: assert-glibc-floor.sh <binary> <max, e.g. 2.28>
set -euo pipefail
bin=$1
max=$2
need=$(readelf -V "$bin" | grep -o 'GLIBC_2\.[0-9]*' | sort -t. -k2 -n | uniq | tail -n1)
echo "$bin requires up to $need (allowed: GLIBC_$max)"
test -n "$need"
test "$(printf '%s\n' "GLIBC_$max" "$need" | sort -t. -k2 -n | tail -n1)" = "GLIBC_$max"
