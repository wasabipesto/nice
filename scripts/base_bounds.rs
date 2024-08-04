#!/usr/bin/env rust-script
//! ```cargo
//! [dependencies]
//! nice_common = { path = "../common" }
//! malachite = { version = "0.4.14" }
//! malachite-nz = { version = "0.4.14", features = ["enable_serde"] }
//! ```

use malachite::natural::Natural;
use malachite::num::arithmetic::traits::Pow;
use malachite::num::conversion::traits::Digits;
use nice_common::base_range::get_base_range_u128;

fn get_sqube_num_digits(num_u128: u128, base: u32) -> u32 {
    let mut num_digits = 0usize;

    // convert u128 to natural
    let num = Natural::from(num_u128);

    // square the number, convert to base and save the digits
    let squared = (&num).pow(2);
    num_digits += squared.to_digits_asc(&base).len();

    // cube, convert to base and save the digits
    let cubed = squared * &num;
    num_digits += cubed.to_digits_asc(&base).len();

    num_digits as u32
}

fn main() {
    for base in 10..=50 {
        match get_base_range_u128(base).unwrap() {
            Some(base_range) => {
                for num in (base_range.range_start - 2)..=(base_range.range_start + 1) {
                    let num_digits = get_sqube_num_digits(num, base);
                    print!("Base {base}, Number {num}, Digits: {num_digits} ");
                    if num == base_range.range_start {
                        print!("***");
                    }
                    println!();
                }
                for num in (base_range.range_end - 1)..=(base_range.range_end + 2) {
                    let num_digits = get_sqube_num_digits(num, base);
                    print!("Base {base}, Number {num}, Digits: {num_digits} ");
                    if num == base_range.range_end {
                        print!("***");
                    }
                    println!();
                }
            }
            None => {
                continue;
            }
        }
    }
}
