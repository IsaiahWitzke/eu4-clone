const FRAC_BITS: i64 = 16;
const SCALE: i64 = 1 << 16;

#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Debug)]
pub struct Fixed {
    raw: i64,
}

impl std::ops::Add for Fixed {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self {
            raw: self.raw + rhs.raw,
        }
    }
}

impl std::ops::Sub for Fixed {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        Self {
            raw: self.raw - rhs.raw,
        }
    }
}

impl std::ops::Mul for Fixed {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        let wide = self.raw as i128 * rhs.raw as i128;
        Self { raw: (wide >> FRAC_BITS) as i64 }
    }
}

impl std::ops::Div for Fixed {
    type Output = Self;

    fn div(self, rhs: Self) -> Self {
        let wide = (self.raw as i128) << FRAC_BITS;
        Self { raw: (wide / (rhs.raw as i128)) as i64 }
    }
}

impl From<i32> for Fixed {
    fn from(value: i32) -> Self {
        return Self {
            raw: (value as i64) << FRAC_BITS,
        };
    }
}

impl TryFrom<&str> for Fixed {
    type Error = std::num::ParseIntError;

    fn try_from(value: &str) -> Result<Self, std::num::ParseIntError> {
        let (sign, digits) = match value.strip_prefix('-') {
            Some(rest) => (-1, rest),
            None => (1, value),
        };

        let (int, fraction, fraction_digits_len) =
            if let Some((int, fraction)) = digits.split_once(".") {
                (
                    int.parse::<i32>()?,
                    fraction.parse::<i32>()?,
                    fraction.len(),
                )
            } else {
                (digits.parse::<i32>()?, 0, 0)
            };
        let int = int << FRAC_BITS;
        let fraction = ((fraction as i64) << FRAC_BITS) / (10 as i64).pow(fraction_digits_len as u32);
        Ok(Fixed {
            raw: sign * (int as i64 + fraction as i64),
        })
    }
}

impl std::fmt::Display for Fixed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // don't care about f64 non-determinism for display purposes
        write!(f, "{:.6}", self.raw as f64 / SCALE as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add() {
        assert_eq!(Fixed::from(4 + 5), Fixed::from(4) + Fixed::from(5));
    }

    #[test]
    fn test_sub() {
        assert_eq!(Fixed::from(4 - 5), Fixed::from(4) - Fixed::from(5));
    }

    #[test]
    fn test_mul() {
        assert_eq!(Fixed::from(4 * 5), Fixed::from(4) * Fixed::from(5));
    }

    #[test]
    fn test_div() {
        assert_eq!(Fixed::from(4), Fixed::from(12) / Fixed::from(3));
    }

    #[test]
    fn test_fmt() {
        assert_eq!(
            "-123.554993",
            format!("{}", Fixed::from(-123_554993) / Fixed::from(1_000000))
        );
    }

    #[test]
    fn test_from_str() {
        assert_eq!(
            Fixed::try_from("-123.555").unwrap(),
            Fixed::from(-123_555) / Fixed::from(1_000)
        );
    }
}

