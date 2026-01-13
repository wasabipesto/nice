// CUDA kernels for nice number checking
// Implements GPU-accelerated versions of the hot loop operations

// Use CUDA built-in types instead of stdint.h
typedef unsigned long long uint64_t;
typedef unsigned int uint32_t;
typedef unsigned char uint8_t;

// ============================================================================
// u128 Arithmetic Helpers
// ============================================================================
// CUDA doesn't have native u128, so we represent it as two u64s
// lo = lower 64 bits, hi = upper 64 bits

struct u128 {
    uint64_t lo;
    uint64_t hi;
    
    __device__ __host__ u128() : lo(0), hi(0) {}
    __device__ __host__ u128(uint64_t l, uint64_t h) : lo(l), hi(h) {}
    __device__ __host__ explicit u128(uint64_t l) : lo(l), hi(0) {}
};

// Addition with carry
__device__ __host__ inline u128 add_u128(const u128& a, const u128& b) {
    u128 result;
    result.lo = a.lo + b.lo;
    result.hi = a.hi + b.hi + (result.lo < a.lo ? 1 : 0); // Add carry
    return result;
}

// Multiplication: u128 = u64 * u64
__device__ __host__ inline u128 mul_u64_u64(uint64_t a, uint64_t b) {
    // Split into 32-bit parts
    uint64_t a_lo = a & 0xFFFFFFFFULL;
    uint64_t a_hi = a >> 32;
    uint64_t b_lo = b & 0xFFFFFFFFULL;
    uint64_t b_hi = b >> 32;
    
    uint64_t p0 = a_lo * b_lo;
    uint64_t p1 = a_lo * b_hi;
    uint64_t p2 = a_hi * b_lo;
    uint64_t p3 = a_hi * b_hi;
    
    uint64_t middle = p1 + p2 + (p0 >> 32);
    
    u128 result;
    result.lo = (middle << 32) | (p0 & 0xFFFFFFFFULL);
    result.hi = p3 + (middle >> 32);
    return result;
}

// Multiplication: u128 * u64 -> u256 (but we only need lower 192 bits for n^3)
// Returns lower 128 bits in result, upper 64 bits in carry
__device__ __host__ inline u128 mul_u128_u64(const u128& a, uint64_t b, uint64_t& carry) {
    u128 lo_prod = mul_u64_u64(a.lo, b);
    u128 hi_prod = mul_u64_u64(a.hi, b);
    
    // Result is: lo_prod + (hi_prod << 64)
    u128 result;
    result.lo = lo_prod.lo;
    result.hi = lo_prod.hi + hi_prod.lo;
    carry = hi_prod.hi + (result.hi < lo_prod.hi ? 1 : 0);
    
    return result;
}

// Extended precision structure for n^2 and n^3 (up to 256 bits)
struct u256 {
    uint64_t limbs[4]; // limbs[0] = lowest 64 bits
    
    __device__ __host__ u256() {
        limbs[0] = limbs[1] = limbs[2] = limbs[3] = 0;
    }
};

// Square a u128 number -> u256
__device__ inline u256 square_u128(const u128& n) {
    u256 result;
    
    // n^2 = (hi * 2^64 + lo)^2 = hi^2 * 2^128 + 2*hi*lo*2^64 + lo^2
    u128 lo_sq = mul_u64_u64(n.lo, n.lo);
    u128 hi_sq = mul_u64_u64(n.hi, n.hi);
    u128 mid = mul_u64_u64(n.lo, n.hi);
    
    // Double the middle term (2*hi*lo)
    uint64_t mid_carry = mid.hi >> 63;
    mid.hi = (mid.hi << 1) | (mid.lo >> 63);
    mid.lo = mid.lo << 1;
    
    // Assemble: lo_sq + (mid << 64) + (hi_sq << 128)
    result.limbs[0] = lo_sq.lo;
    result.limbs[1] = lo_sq.hi + mid.lo;
    uint64_t carry = (result.limbs[1] < lo_sq.hi) ? 1 : 0;
    result.limbs[2] = mid.hi + carry + hi_sq.lo;
    carry = (result.limbs[2] < mid.hi + carry) ? 1 : 0;
    result.limbs[3] = hi_sq.hi + mid_carry + carry;
    
    return result;
}

// Cube a u128 number -> u256 (actually produces more, but we store lower 256 bits)
__device__ inline u256 cube_u128(const u128& n) {
    // n^3 = n * n^2
    u256 n_squared = square_u128(n);
    u256 result;
    
    // Multiply n_squared by n
    uint64_t carry = 0;
    for (int i = 0; i < 4; i++) {
        // Multiply limb by n.lo
        u128 prod_lo = mul_u64_u64(n_squared.limbs[i], n.lo);
        uint64_t sum_lo = prod_lo.lo + carry;
        carry = prod_lo.hi + (sum_lo < prod_lo.lo ? 1 : 0);
        
        // Multiply limb by n.hi (shifted by 64 bits)
        if (i < 3) {
            u128 prod_hi = mul_u64_u64(n_squared.limbs[i], n.hi);
            uint64_t sum = carry + prod_hi.lo;
            carry = prod_hi.hi + (sum < carry ? 1 : 0);
            result.limbs[i] = sum_lo;
            carry = sum;
        } else {
            result.limbs[i] = sum_lo;
        }
    }
    
    return result;
}

// Division by base (for base conversion)
// Divides u256 by base, returns remainder
// Uses long division: for each limb, divide (remainder * 2^64 + limb) by base
__device__ inline uint32_t div_u256_by_u32(u256& n, uint32_t base) {
    uint64_t remainder = 0;
    
    // Long division from most significant to least significant limb
    for (int i = 3; i >= 0; i--) {
        // We need to divide (remainder << 64 | n.limbs[i]) by base
        // Since remainder < base and base fits in 32 bits, we can split this:
        // Split the limb into two 32-bit halves for easier processing
        uint32_t limb_hi = (uint32_t)(n.limbs[i] >> 32);
        uint32_t limb_lo = (uint32_t)(n.limbs[i] & 0xFFFFFFFFULL);
        
        // Divide the high part: (remainder << 32 | limb_hi) / base
        uint64_t div_hi = (remainder << 32) | limb_hi;
        uint64_t quot_hi = div_hi / base;
        remainder = div_hi % base;
        
        // Divide the low part: (remainder << 32 | limb_lo) / base
        uint64_t div_lo = (remainder << 32) | limb_lo;
        uint64_t quot_lo = div_lo / base;
        remainder = div_lo % base;
        
        // Combine quotients
        n.limbs[i] = (quot_hi << 32) | quot_lo;
    }
    
    return (uint32_t)remainder;
}

// Check if u256 is zero
__device__ inline bool is_zero_u256(const u256& n) {
    return (n.limbs[0] | n.limbs[1] | n.limbs[2] | n.limbs[3]) == 0;
}

// ============================================================================
// Kernel 1: Count unique digits (detailed mode)
// ============================================================================

extern "C" __global__ void count_unique_digits_kernel(
    const uint64_t* numbers_lo,      // Lower 64 bits of input numbers
    const uint64_t* numbers_hi,      // Upper 64 bits of input numbers
    uint32_t* unique_counts,         // Output: number of unique digits for each
    const uint32_t base,
    const size_t n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    u128 num(numbers_lo[idx], numbers_hi[idx]);
    
    // Calculate n^2 and n^3
    u256 squared = square_u128(num);
    u256 cubed = cube_u128(num);
    
    // Track unique digits using a bitmask (up to 128 bits for bases up to 128)
    // Use two uint64_t to represent 128 bits
    uint64_t digits_lo = 0;  // Bits 0-63
    uint64_t digits_hi = 0;  // Bits 64-127
    
    // Process squared: convert to base and mark digits
    u256 temp_squared = squared;
    while (!is_zero_u256(temp_squared)) {
        uint32_t digit = div_u256_by_u32(temp_squared, base);
        if (digit < 64) {
            digits_lo |= (1ULL << digit);
        } else {
            digits_hi |= (1ULL << (digit - 64));
        }
    }
    
    // Process cubed: convert to base and mark digits
    u256 temp_cubed = cubed;
    while (!is_zero_u256(temp_cubed)) {
        uint32_t digit = div_u256_by_u32(temp_cubed, base);
        if (digit < 64) {
            digits_lo |= (1ULL << digit);
        } else {
            digits_hi |= (1ULL << (digit - 64));
        }
    }
    
    // Count the number of set bits
    unique_counts[idx] = __popcll(digits_lo) + __popcll(digits_hi);
}

// ============================================================================
// Kernel 2: Check if nice (niceonly mode - optimized)
// ============================================================================

extern "C" __global__ void check_is_nice_kernel(
    const uint64_t* numbers_lo,      // Lower 64 bits of input numbers
    const uint64_t* numbers_hi,      // Upper 64 bits of input numbers
    uint8_t* is_nice,                // Output: 1 if nice, 0 otherwise
    const uint32_t base,
    const size_t n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    u128 num(numbers_lo[idx], numbers_hi[idx]);
    
    // Calculate n^2 and n^3
    u256 squared = square_u128(num);
    u256 cubed = cube_u128(num);
    
    // Track unique digits with early exit on duplicates
    uint64_t digits_lo = 0;  // Bits 0-63
    uint64_t digits_hi = 0;  // Bits 64-127
    
    // Process squared: convert to base and check for duplicates
    u256 temp_squared = squared;
    while (!is_zero_u256(temp_squared)) {
        uint32_t digit = div_u256_by_u32(temp_squared, base);
        
        // Early exit if we've seen this digit before
        if (digit < 64) {
            if (digits_lo & (1ULL << digit)) {
                is_nice[idx] = 0;
                return;
            }
            digits_lo |= (1ULL << digit);
        } else {
            if (digits_hi & (1ULL << (digit - 64))) {
                is_nice[idx] = 0;
                return;
            }
            digits_hi |= (1ULL << (digit - 64));
        }
    }
    
    // Process cubed: convert to base and check for duplicates
    u256 temp_cubed = cubed;
    while (!is_zero_u256(temp_cubed)) {
        uint32_t digit = div_u256_by_u32(temp_cubed, base);
        
        // Early exit if we've seen this digit before
        if (digit < 64) {
            if (digits_lo & (1ULL << digit)) {
                is_nice[idx] = 0;
                return;
            }
            digits_lo |= (1ULL << digit);
        } else {
            if (digits_hi & (1ULL << (digit - 64))) {
                is_nice[idx] = 0;
                return;
            }
            digits_hi |= (1ULL << (digit - 64));
        }
    }
    
    // Check if we have exactly 'base' unique digits
    int count = __popcll(digits_lo) + __popcll(digits_hi);
    is_nice[idx] = (count == base) ? 1 : 0;
}

// ============================================================================
// Kernel 3: Residue filter (preprocessing)
// ============================================================================

extern "C" __global__ void filter_by_residue_kernel(
    const uint64_t* numbers_lo,      // Input: lower 64 bits
    const uint64_t* numbers_hi,      // Input: upper 64 bits
    const uint64_t* filter_residues, // Residue filter array
    const size_t filter_size,
    uint8_t* matches,                // Output: 1 if passes filter, 0 otherwise
    const uint32_t base,
    const size_t n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    // Calculate num % (base - 1)
    // For u128, we need to do this carefully
    uint64_t num_lo = numbers_lo[idx];
    uint64_t num_hi = numbers_hi[idx];
    
    uint64_t base_minus_one = base - 1;
    
    // Compute (hi * 2^64 + lo) % (base-1)
    // = ((hi % (base-1)) * (2^64 % (base-1)) + (lo % (base-1))) % (base-1)
    uint64_t hi_mod = num_hi % base_minus_one;
    uint64_t lo_mod = num_lo % base_minus_one;
    
    // 2^64 % (base-1) can be precomputed, but for simplicity:
    uint64_t power_mod = (1ULL << 63) % base_minus_one;
    power_mod = (power_mod + power_mod) % base_minus_one;
    
    uint64_t residue = (hi_mod * power_mod + lo_mod) % base_minus_one;
    
    // Check if residue is in the filter
    matches[idx] = 0;
    for (size_t i = 0; i < filter_size; i++) {
        if (residue == filter_residues[i]) {
            matches[idx] = 1;
            break;
        }
    }
}