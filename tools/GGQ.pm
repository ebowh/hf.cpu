# GGQ.pm - dequantizers for a few ggml block types in plain Perl (used by tools/ref-forward.pl
# and tools/mkmodel.pl). Independent of the C implementation: written from the block layouts.
package GGQ;
use strict;
use warnings;
use Exporter 'import';
our @EXPORT_OK = qw(half2f type_info dequant_blocks);

sub half2f {
    my $h = shift; my $s = ($h & 0x8000) ? -1 : 1; my $e = ($h >> 10) & 31; my $m = $h & 1023;
    return $s * $m * 2**-24 if $e == 0;
    return $s * (1 + $m / 1024) * 2**($e - 15);
}

# type id => [block elements, block bytes]
my %INFO = (0 => [1, 4], 1 => [1, 2], 2 => [32, 18], 6 => [32, 22], 8 => [32, 34],
            12 => [256, 144], 13 => [256, 176], 14 => [256, 210]);
sub type_info { return $INFO{$_[0]} }

sub scale_min_k4 {
    my ($j, @q) = @_;
    return ($q[$j] & 63, $q[$j + 4] & 63) if $j < 4;
    return (($q[$j + 4] & 0xF) | (($q[$j - 4] >> 6) << 4), ($q[$j + 4] >> 4) | (($q[$j] >> 6) << 4));
}

# Dequantize a buffer of whole blocks of the given type into a list of doubles.
sub dequant_blocks {
    my ($type, $buf) = @_;
    my ($qk, $bs) = @{ $INFO{$type} or die "GGQ: unsupported type $type\n" };
    my @out;
    if ($type == 0) { return [ unpack('f<*', $buf) ]; }
    if ($type == 1) { return [ map { half2f($_) } unpack('v*', $buf) ]; }
    for (my $o = 0; $o < length $buf; $o += $bs) {
        my $b = substr($buf, $o, $bs);
        if ($type == 8) {
            my $d = half2f(unpack('v', $b)); push @out, map { $_ * $d } unpack('c32', substr($b, 2));
        } elsif ($type == 2) {
            my $d = half2f(unpack('v', $b)); my @q = unpack('C16', substr($b, 2));
            push @out, map { (($_ & 0xF) - 8) * $d } @q; push @out, map { (($_ >> 4) - 8) * $d } @q;
        } elsif ($type == 6) {
            my $d = half2f(unpack('v', $b)); my $qh = unpack('V', substr($b, 2, 4)); my @q = unpack('C16', substr($b, 6));
            push @out, map { ((($q[$_] & 0xF) | ((($qh >> $_) & 1) << 4)) - 16) * $d } 0 .. 15;
            push @out, map { ((($q[$_] >> 4) | ((($qh >> ($_ + 16)) & 1) << 4)) - 16) * $d } 0 .. 15;
        } elsif ($type == 12 || $type == 13) {
            my ($d, $dmin) = map { half2f($_) } unpack('v2', $b);
            my @sc = unpack('C12', substr($b, 4, 12));
            my @qh = $type == 13 ? unpack('C32', substr($b, 16, 32)) : ();
            my @ql = unpack($type == 13 ? 'C128' : 'C128', substr($b, $type == 13 ? 48 : 16));
            my ($is, $u1, $u2) = (0, 1, 2);
            for my $j (0 .. 3) {
                my ($s1, $m1) = scale_min_k4($is, @sc); my ($s2, $m2) = scale_min_k4($is + 1, @sc);
                my @q = @ql[32 * $j .. 32 * $j + 31];
                if ($type == 12) {
                    push @out, map { $d * $s1 * ($_ & 0xF) - $dmin * $m1 } @q;
                    push @out, map { $d * $s2 * ($_ >> 4) - $dmin * $m2 } @q;
                } else {
                    push @out, map { $d * $s1 * (($q[$_] & 0xF) + (($qh[$_] & $u1) ? 16 : 0)) - $dmin * $m1 } 0 .. 31;
                    push @out, map { $d * $s2 * (($q[$_] >> 4) + (($qh[$_] & $u2) ? 16 : 0)) - $dmin * $m2 } 0 .. 31;
                    $u1 <<= 2; $u2 <<= 2;
                }
                $is += 2;
            }
        } elsif ($type == 14) {
            my @ql = unpack('C128', $b); my @qh = unpack('C64', substr($b, 128)); my @sc = unpack('c16', substr($b, 192));
            my $d = half2f(unpack('v', substr($b, 208, 2)));
            my @y = (0) x 256;
            for my $n (0 .. 1) {
                for my $l (0 .. 31) {
                    my $is = int($l / 16); my $qlo = 64 * $n; my $qhi = 32 * $n; my $sb = 8 * $n; my $yb = 128 * $n;
                    my $h = $qh[$qhi + $l];
                    my $q1 = (($ql[$qlo + $l] & 0xF) | ((($h >> 0) & 3) << 4)) - 32;
                    my $q2 = (($ql[$qlo + $l + 32] & 0xF) | ((($h >> 2) & 3) << 4)) - 32;
                    my $q3 = (($ql[$qlo + $l] >> 4) | ((($h >> 4) & 3) << 4)) - 32;
                    my $q4 = (($ql[$qlo + $l + 32] >> 4) | ((($h >> 6) & 3) << 4)) - 32;
                    $y[$yb + $l] = $d * $sc[$sb + $is] * $q1;      $y[$yb + $l + 32] = $d * $sc[$sb + $is + 2] * $q2;
                    $y[$yb + $l + 64] = $d * $sc[$sb + $is + 4] * $q3; $y[$yb + $l + 96] = $d * $sc[$sb + $is + 6] * $q4;
                }
            }
            push @out, @y;
        }
    }
    return \@out;
}
1;
