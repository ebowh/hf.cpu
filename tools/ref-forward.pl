#!/usr/bin/perl
# ref-forward.pl - independent, plain-Perl, double-precision forward pass for the
# tiny models written by mkmodel.pl (and any small qwen2/qwen3/llama GGUF).
# Used as an oracle for the C engine.
#
#   perl tools/ref-forward.pl MODEL.gguf "id id id ..." [--steps N]
# Prints, for each generated step: "gen POS ID LOGPROB top: id:lp ..." (greedy).
use strict;
use warnings;
use Getopt::Long;
use FindBin qw($Bin);
use lib $Bin;
use GGQ qw(dequant_blocks type_info);

my $steps = 4;
GetOptions('steps=i' => \$steps) or die;
my ($path, $idlist) = @ARGV;
die "usage: $0 MODEL.gguf \"ids\" [--steps N]\n" unless defined $idlist;
my @prompt = split /[ ,]+/, $idlist;

open(my $fh, '<:raw', $path) or die "cannot open $path\n";
local $/; my $data = <$fh>; close $fh;
my $pos = 4;
sub u32 { my $v = unpack('V', substr($data, $pos, 4)); $pos += 4; $v }
sub u64 { my $v = unpack('Q<', substr($data, $pos, 8)); $pos += 8; $v }
sub str { my $n = u64(); my $s = substr($data, $pos, $n); $pos += $n; $s }
die "bad magic\n" unless substr($data, 0, 4) eq 'GGUF';
u32(); my $nt = u64(); my $nkv = u64();
my %kv;
for (1 .. $nkv) {
    my $k = str(); my $t = u32();
    if ($t == 8) { $kv{$k} = str(); }
    elsif ($t == 4) { $kv{$k} = u32(); }
    elsif ($t == 6) { $kv{$k} = unpack('f<', substr($data, $pos, 4)); $pos += 4; }
    else { die "unsupported kv type $t\n"; }
}
my %T;
for (1 .. $nt) {
    my $name = str(); my $nd = u32(); my @ne = map { u64() } 1 .. $nd; my $type = u32(); my $off = u64();
    $T{$name} = { ne => \@ne, type => $type, off => $off };
}
my $align = 32;
my $dstart = int(($pos + $align - 1) / $align) * $align;

sub half2f {
    my $h = shift; my $s = ($h & 0x8000) ? -1 : 1; my $e = ($h >> 10) & 31; my $m = $h & 1023;
    return $s * $m * 2**-24 if $e == 0;
    return $s * (1 + $m / 1024) * 2**($e - 15);
}
# Dequantize a whole tensor to a flat array of doubles
sub load_tensor {
    my $t = $T{$_[0]} or die "missing tensor $_[0]\n";
    my $n = 1; $n *= $_ for @{$t->{ne}};
    my $o = $dstart + $t->{off};
    my $ti = type_info($t->{type}) or die "unsupported tensor type $t->{type}\n";
    my $bytes = $n / $ti->[0] * $ti->[1];
    return dequant_blocks($t->{type}, substr($data, $o, $bytes));
}

my $arch = $kv{'general.architecture'};
my $L = $kv{"$arch.block_count"}; my $E = $kv{"$arch.embedding_length"}; my $FF = $kv{"$arch.feed_forward_length"};
my $H = $kv{"$arch.attention.head_count"}; my $HKV = $kv{"$arch.attention.head_count_kv"};
my $eps = $kv{"$arch.attention.layer_norm_rms_epsilon"}; my $base = $kv{"$arch.rope.freq_base"};
my $HD = $E / $H; my $neox = $arch ne 'llama';
my $emb = load_tensor('token_embd.weight'); my $V = @$emb / $E;
my $outnorm = load_tensor('output_norm.weight');
my $outw = $T{'output.weight'} ? load_tensor('output.weight') : $emb;
my @lay;
for my $l (0 .. $L - 1) {
    my %w;
    for my $n (qw(attn_norm ffn_norm attn_q attn_k attn_v attn_output ffn_gate ffn_up ffn_down)) { $w{$n} = load_tensor("blk.$l.$n.weight"); }
    for my $n (qw(attn_q attn_k attn_v)) { $w{"b$n"} = load_tensor("blk.$l.$n.bias") if $T{"blk.$l.$n.bias"}; }
    for my $n (qw(attn_q_norm attn_k_norm)) { $w{$n} = load_tensor("blk.$l.$n.weight") if $T{"blk.$l.$n.weight"}; }
    push @lay, \%w;
}

sub matvec {            # W: flat [rows][cols], x: [cols] -> [rows]
    my ($W, $x, $rows, $cols, $b) = @_;
    my @y;
    for my $r (0 .. $rows - 1) {
        my $s = $b ? $b->[$r] : 0; my $o = $r * $cols;
        for my $c (0 .. $cols - 1) { $s += $W->[$o + $c] * $x->[$c]; }
        push @y, $s;
    }
    return \@y;
}
sub rmsnorm {
    my ($x, $w) = @_;
    my $s = 0; $s += $_ * $_ for @$x; my $sc = 1 / sqrt($s / @$x + $eps);
    return [ map { $x->[$_] * $sc * ($w ? $w->[$_] : 1) } 0 .. $#$x ];
}
sub rope {              # in place on a vector made of $nh heads
    my ($v, $nh, $p) = @_;
    my $half = $HD / 2;
    for my $h (0 .. $nh - 1) {
        for my $i (0 .. $half - 1) {
            my $th = $p * $base ** (-2.0 * $i / $HD);
            my ($c, $s) = (cos($th), sin($th));
            my ($a, $b) = $neox ? ($h * $HD + $i, $h * $HD + $i + $half) : ($h * $HD + 2 * $i, $h * $HD + 2 * $i + 1);
            my ($x0, $x1) = ($v->[$a], $v->[$b]);
            $v->[$a] = $x0 * $c - $x1 * $s; $v->[$b] = $x0 * $s + $x1 * $c;
        }
    }
}

my (@K, @Vv);           # per layer: list of position vectors
for my $l (0 .. $L - 1) { $K[$l] = []; $Vv[$l] = []; }

sub forward_token {
    my ($tok, $p) = @_;
    my $x = [ @$emb[$tok * $E .. $tok * $E + $E - 1] ];
    for my $l (0 .. $L - 1) {
        my $w = $lay[$l];
        my $xn = rmsnorm($x, $w->{attn_norm});
        my $q = matvec($w->{attn_q}, $xn, $H * $HD, $E, $w->{battn_q});
        my $k = matvec($w->{attn_k}, $xn, $HKV * $HD, $E, $w->{battn_k});
        my $v = matvec($w->{attn_v}, $xn, $HKV * $HD, $E, $w->{battn_v});
        if ($w->{attn_q_norm}) {
            for my $h (0 .. $H - 1) { my $n = rmsnorm([ @$q[$h * $HD .. $h * $HD + $HD - 1] ], $w->{attn_q_norm}); @$q[$h * $HD .. $h * $HD + $HD - 1] = @$n; }
            for my $h (0 .. $HKV - 1) { my $n = rmsnorm([ @$k[$h * $HD .. $h * $HD + $HD - 1] ], $w->{attn_k_norm}); @$k[$h * $HD .. $h * $HD + $HD - 1] = @$n; }
        }
        rope($q, $H, $p); rope($k, $HKV, $p);
        push @{$K[$l]}, $k; push @{$Vv[$l]}, $v;
        my @att = (0) x ($H * $HD);
        my $grp = $H / $HKV;
        for my $h (0 .. $H - 1) {
            my $kvh = int($h / $grp);
            my @s;
            for my $j (0 .. $p) {
                my $d = 0; my $kk = $K[$l][$j];
                $d += $q->[$h * $HD + $_] * $kk->[$kvh * $HD + $_] for 0 .. $HD - 1;
                push @s, $d / sqrt($HD);
            }
            my $mx = $s[0]; for (@s) { $mx = $_ if $_ > $mx; }
            my $sum = 0; @s = map { my $e = exp($_ - $mx); $sum += $e; $e } @s;
            for my $j (0 .. $p) {
                my $wgt = $s[$j] / $sum; my $vv = $Vv[$l][$j];
                $att[$h * $HD + $_] += $wgt * $vv->[$kvh * $HD + $_] for 0 .. $HD - 1;
            }
        }
        my $o = matvec($w->{attn_output}, \@att, $E, $H * $HD);
        $x->[$_] += $o->[$_] for 0 .. $E - 1;
        my $xn2 = rmsnorm($x, $w->{ffn_norm});
        my $g = matvec($w->{ffn_gate}, $xn2, $FF, $E);
        my $u = matvec($w->{ffn_up}, $xn2, $FF, $E);
        my @hid = map { ($g->[$_] / (1 + exp(-$g->[$_]))) * $u->[$_] } 0 .. $FF - 1;
        my $dn = matvec($w->{ffn_down}, \@hid, $E, $FF);
        $x->[$_] += $dn->[$_] for 0 .. $E - 1;
    }
    return matvec($outw, rmsnorm($x, $outnorm), $V, $E);
}

my $p = 0; my $logits;
for my $t (@prompt) { $logits = forward_token($t, $p++); }
for my $step (0 .. $steps - 1) {
    my $mx = $logits->[0]; my $best = 0;
    for my $i (1 .. $#$logits) { if ($logits->[$i] > $mx) { $mx = $logits->[$i]; $best = $i; } }
    my $sum = 0; $sum += exp($_ - $mx) for @$logits; my $lse = log($sum);
    my @order = sort { $logits->[$b] <=> $logits->[$a] || $a <=> $b } 0 .. $#$logits;
    printf "gen %d %d %.6f top: %s\n", $step, $best, $logits->[$best] - $mx - $lse,
        join(' ', map { sprintf('%d:%.6f', $_, $logits->[$_] - $mx - $lse) } @order[0 .. 4]);
    last if $step == $steps - 1;
    $logits = forward_token($best, $p++);
}
