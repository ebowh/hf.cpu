#!/usr/bin/perl
# mkmodel.pl - write a tiny random-weight transformer GGUF (no tokenizer) for tests.
#
#   perl tools/mkmodel.pl OUT.gguf [--arch qwen2|qwen3|llama] [--wtype f32|f16|q8_0]
#                         [--layers 2] [--vocab 300] [--seed 1] [--size tiny|big] [--deq OUT]
# Dimensions: embed 64, 4 heads / 2 KV heads (head_dim 16), FFN 128.
# --size big: embed 512, 8 heads (2 KV, head_dim 64), FFN 2048, for speed tests.use strict;
use warnings;
use Getopt::Long;

my ($arch, $wtype, $layers, $vocab, $seed, $deq, $size) = ('qwen2', 'f32', 2, 300, 1, undef, 'tiny');
GetOptions('size=s' => \$size, 'deq=s' => \$deq, 'arch=s' => \$arch, 'wtype=s' => \$wtype, 'layers=i' => \$layers, 'vocab=i' => \$vocab, 'seed=i' => \$seed)
    or die "bad options\n";
my $out = shift or die "usage: $0 OUT.gguf [options]\n";
die "unknown arch\n" unless $arch =~ /^(qwen2|qwen3|llama)$/;
die "unknown wtype\n" unless $wtype =~ /^(f32|f16|q8_0)$/;
srand($seed);

my ($E, $H, $HKV, $HD, $FF) = $size eq 'big' ? (512, 8, 2, 64, 2048) : (64, 4, 2, 16, 128);
my %TYPEID = (f32 => 0, f16 => 1, q8_0 => 8);

sub u32 { pack('V', $_[0]) }
sub u64 { pack('Q<', $_[0]) }
sub str { u64(length $_[0]) . $_[0] }
sub kv_str { str($_[0]) . u32(8) . str($_[1]) }
sub kv_u32 { str($_[0]) . u32(4) . u32($_[1]) }
sub kv_f32 { str($_[0]) . u32(6) . pack('f<', $_[1]) }

sub f2h {                      # float -> IEEE half bits (round to nearest, no subnormal care needed here)
    my $f = shift;
    my $x = unpack('L', pack('f', $f));
    my $s = ($x >> 16) & 0x8000; my $e = (($x >> 23) & 0xff) - 127 + 15; my $m = $x & 0x7fffff;
    return $s if $e <= 0;
    return $s | 0x7c00 if $e >= 31;
    my $h = ($e << 10) | ($m >> 13);
    $h++ if ($m & 0x1fff) > 0x1000;
    return $s | $h;
}
sub gauss { return (rand() * 2 - 1) * 1.7 if $size eq 'big'; my $s = 0; $s += rand() for 1 .. 6; return ($s - 3.0) / 0.7071; }   # big models: uniform noise (faster)   # ~N(0,1)

sub encode {
    my ($vals, $deqout) = @_;   # flat list of floats; $deqout receives the values the file really stores
    if ($deqout) { @$deqout = @$vals; }
    return pack('f<*', @$vals) if $wtype eq 'f32';
    return pack('v*', map { f2h($_) } @$vals) if $wtype eq 'f16';
    my $buf = '';
    for (my $i = 0; $i < @$vals; $i += 32) {
        my @b = @$vals[$i .. $i + 31];
        my $amax = 0; for (@b) { my $a = abs; $amax = $a if $a > $amax; }
        my $d = $amax / 127; my $dh = f2h($d); my $id = $d ? 1 / $d : 0;
        my @q = map { int($_ * $id + ($_ * $id < 0 ? -0.5 : 0.5)) } @b;
        $buf .= pack('v', $dh) . pack('c32', @q);
        if ($deqout) { my $dh_val = ($dh & 0x3ff) / 1024; my $e = ($dh >> 10) & 31; my $dv = $e ? (1 + $dh_val) * 2**($e - 15) : $dh_val * 2**-14; @$deqout[$i .. $i + 31] = map { $_ * $dv } @q; }
    }
    return $buf;
}

my (@tens, @tens_deq);
sub mat { my ($name, $cols, $rows, $scale) = @_;
    my @v = map { gauss() * $scale } 1 .. $cols * $rows;
    my @dq;
    push @tens, [$name, [$cols, $rows], $TYPEID{$wtype}, encode(\@v, \@dq)];
    push @tens_deq, [$name, [$cols, $rows], 0, pack('f<*', @dq)];
}
sub vecw { my ($name, $n, $base, $noise) = @_;
    my $bytes = pack('f<*', map { $base + gauss() * $noise } 1 .. $n);
    push @tens, [$name, [$n], 0, $bytes];
    push @tens_deq, [$name, [$n], 0, $bytes];
}

mat('token_embd.weight', $E, $vocab, 0.5);
vecw('output_norm.weight', $E, 1.0, 0.1);
for my $l (0 .. $layers - 1) {
    vecw("blk.$l.attn_norm.weight", $E, 1.0, 0.1);
    mat("blk.$l.attn_q.weight", $E, $H * $HD, 1.2 / sqrt($E));
    mat("blk.$l.attn_k.weight", $E, $HKV * $HD, 1.2 / sqrt($E));
    mat("blk.$l.attn_v.weight", $E, $HKV * $HD, 1.0 / sqrt($E));
    mat("blk.$l.attn_output.weight", $H * $HD, $E, 1.0 / sqrt($H * $HD));
    if ($arch eq 'qwen2') {
        vecw("blk.$l.attn_q.bias", $H * $HD, 0, 0.3); vecw("blk.$l.attn_k.bias", $HKV * $HD, 0, 0.3); vecw("blk.$l.attn_v.bias", $HKV * $HD, 0, 0.2);
    }
    if ($arch eq 'qwen3') { vecw("blk.$l.attn_q_norm.weight", $HD, 1.0, 0.1); vecw("blk.$l.attn_k_norm.weight", $HD, 1.0, 0.1); }
    vecw("blk.$l.ffn_norm.weight", $E, 1.0, 0.1);
    mat("blk.$l.ffn_gate.weight", $E, $FF, 1.2 / sqrt($E));
    mat("blk.$l.ffn_up.weight", $E, $FF, 1.0 / sqrt($E));
    mat("blk.$l.ffn_down.weight", $FF, $E, 1.0 / sqrt($FF));
}

my $align = 32;
my $kvs = kv_str('general.architecture', $arch) . kv_str('general.name', "tiny $arch $wtype")
        . kv_u32("$arch.block_count", $layers) . kv_u32("$arch.context_length", 512)
        . kv_u32("$arch.embedding_length", $E) . kv_u32("$arch.feed_forward_length", $FF)
        . kv_u32("$arch.attention.head_count", $H) . kv_u32("$arch.attention.head_count_kv", $HKV)
        . kv_f32("$arch.attention.layer_norm_rms_epsilon", 1e-6) . kv_f32("$arch.rope.freq_base", 10000.0);
my $nkv = 10;
my ($infos, $blob) = ('', '');
for my $t (@tens) {
    my ($name, $dims, $id, $data) = @$t;
    $blob .= "\0" x ((-length $blob) % $align);
    $infos .= str($name) . u32(scalar @$dims) . join('', map { u64($_) } @$dims) . u32($id) . u64(length $blob);
    $blob .= $data;
}
my $head = 'GGUF' . u32(3) . u64(scalar @tens) . u64($nkv) . $kvs . $infos;
$head .= "\0" x ((-length $head) % $align);
open(my $fh, '>:raw', $out) or die "cannot write $out: $!\n";
print $fh $head, $blob;
close $fh;
printf "wrote %s (%s, %s): %d tensors, %d bytes\n", $out, $arch, $wtype, scalar @tens, length($head) + length($blob);

if (defined $deq) {
    # the same model with every weight stored as f32 holding exactly the dequantized values
    my ($i2, $b2) = ('', '');
    for my $t (@tens_deq) {
        my ($name, $dims, $id, $data) = @$t;
        $b2 .= "\0" x ((-length $b2) % $align);
        $i2 .= str($name) . u32(scalar @$dims) . join('', map { u64($_) } @$dims) . u32($id) . u64(length $b2);
        $b2 .= $data;
    }
    my $h2 = 'GGUF' . u32(3) . u64(scalar @tens_deq) . u64($nkv) . $kvs . $i2;
    $h2 .= "\0" x ((-length $h2) % $align);
    open(my $f2, '>:raw', $deq) or die "cannot write $deq: $!\n";
    print $f2 $h2, $b2;
    close $f2;
}
