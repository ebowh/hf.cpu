#!/usr/bin/perl
# mkgguf.pl - write a synthetic GGUF file with random tensor data, for tests.
#
#   perl tools/mkgguf.pl OUT.gguf [--arch qwen2] [--layers 2] [--seed 1]
#
# Tensors cover many quantization types so the readers and dequantizers can
# be exercised without downloading a model. Content is random bytes with the
# fp16 scale fields set to small finite values.
use strict;
use warnings;
use Getopt::Long;

my ($arch, $layers, $seed) = ('qwen2', 2, 1);
GetOptions('arch=s' => \$arch, 'layers=i' => \$layers, 'seed=i' => \$seed) or die "bad options\n";
my $out = shift or die "usage: $0 OUT.gguf [--arch A] [--layers N] [--seed S]\n";
srand($seed);

my %T = (f32 => [0, 1, 4], f16 => [1, 1, 2], q4_0 => [2, 32, 18], q4_1 => [3, 32, 20], q5_0 => [6, 32, 22],
         q5_1 => [7, 32, 24], q8_0 => [8, 32, 34], q2_K => [10, 256, 84], q3_K => [11, 256, 110],
         q4_K => [12, 256, 144], q5_K => [13, 256, 176], q6_K => [14, 256, 210], iq4_nl => [20, 32, 18],
         iq4_xs => [23, 256, 136], bf16 => [30, 1, 2], q2_0 => [42, 64, 18]);

sub u32 { pack('V', $_[0]) }
sub u64 { pack('Q<', $_[0]) }
sub str { u64(length $_[0]) . $_[0] }
sub kv_str { str($_[0]) . u32(8) . str($_[1]) }
sub kv_u32 { str($_[0]) . u32(4) . u32($_[1]) }
sub kv_f32 { str($_[0]) . u32(6) . pack('f<', $_[1]) }
sub kv_arr_str { my ($k, @v) = @_; str($k) . u32(9) . u32(8) . u64(scalar @v) . join('', map { str($_) } @v) }

# f16 for 0.01 .. 0.5, little endian
sub rand_half {
    my $f = 0.01 + rand(0.49);
    my $e = int(log($f) / log(2));
    my $m = int((($f / 2**$e) - 1) * 1024);
    return pack('v', (($e + 15) << 10) | $m);
}

sub block { my ($type, $n) = @_; my ($id, $blck, $bytes) = @{$T{$type}};
    my $buf = '';
    for (1 .. $n) {
        my $b = join('', map { chr(int(rand(256))) } 1 .. $bytes);
        # keep the fp16 scale fields finite: first two bytes (d) for most types
        if ($type =~ /^(q4_0|q4_1|q5_0|q5_1|q8_0|iq4_nl|q2_0|q4_K|q5_K|iq4_xs)$/) {
            substr($b, 0, 2) = rand_half();
            substr($b, 2, 2) = rand_half() if $type =~ /^(q4_1|q5_1|q4_K|q5_K)$/;
        }
        if ($type eq 'q6_K') { substr($b, 208, 2) = rand_half(); }
        if ($type eq 'q2_K') { substr($b, 80, 2) = rand_half(); substr($b, 82, 2) = rand_half(); }
        if ($type eq 'q3_K') { substr($b, 108, 2) = rand_half(); }
        $buf .= $b;
    }
    return $buf;
}

my @tens;     # [name, [dims], type, data]
sub add { my ($name, $cols, $rows, $type) = @_;
    my ($id, $blck, $bytes) = @{$T{$type}};
    die "cols must be a multiple of $blck" if $cols % $blck;
    my $data;
    if ($type eq 'f32') { $data = pack('f<*', map { rand(2) - 1 } 1 .. $cols * $rows); }
    elsif ($type eq 'f16') { $data = join('', map { rand_half() } 1 .. $cols * $rows); }
    elsif ($type eq 'bf16') { $data = pack('v*', map { 0x3c00 + int(rand(0x300)) } 1 .. $cols * $rows); }
    else { $data = block($type, ($cols / $blck) * $rows); }
    push @tens, [$name, [$cols, $rows], $id, $data];
}

add('token_embd.weight', 256, 8, 'q8_0');
add('output_norm.weight', 256, 1, 'f32');
my @cycle = qw(q4_K q6_K q5_K q4_0 q5_0 q4_1 q5_1 q3_K q2_K iq4_nl iq4_xs q2_0 f16 bf16);
for my $l (0 .. $layers - 1) {
    my $i = 0;
    for my $n (qw(attn_q attn_k attn_v attn_output ffn_gate ffn_up ffn_down)) {
        add("blk.$l.$n.weight", 256, 4, $cycle[($l * 7 + $i++) % @cycle]);
    }
    add("blk.$l.attn_norm.weight", 256, 1, 'f32');
}

my $align = 32;
my $hdr = 'GGUF' . u32(3) . u64(scalar @tens) . u64(6);
my $kvs = kv_str('general.architecture', $arch) . kv_str('general.name', 'mkgguf synthetic')
        . kv_u32("$arch.block_count", $layers) . kv_u32("$arch.embedding_length", 256)
        . kv_f32("$arch.rope.freq_base", 1000000.0) . kv_arr_str('tokenizer.ggml.tokens', '<s>', 'a', 'b');
my ($infos, $blob) = ('', '');
for my $t (@tens) {
    my ($name, $dims, $id, $data) = @$t;
    $blob .= "\0" x ((-length $blob) % $align);
    $infos .= str($name) . u32(scalar @$dims) . join('', map { u64($_) } @$dims) . u32($id) . u64(length $blob);
    $blob .= $data;
}
my $head = $hdr . $kvs . $infos;
$head .= "\0" x ((-length $head) % $align);
open(my $fh, '>:raw', $out) or die "cannot write $out: $!\n";
print $fh $head, $blob;
close $fh;
printf "wrote %s: %d tensors, %d bytes\n", $out, scalar @tens, length($head) + length($blob);
