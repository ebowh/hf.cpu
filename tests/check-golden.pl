#!/usr/bin/perl
# check-golden.pl - compare hfcpu with llama.cpp reference output recorded by tools/golden.pl.
#
#   perl tests/check-golden.pl MODEL.gguf tests/golden/FILE.golden [--bin ./hfcpu] [--mode ids|text|both] [--tol 0.05]
#
# For each recorded prompt it runs greedy generation with top-5 logprobs and reports:
#   - how many generated tokens agree with llama.cpp before the first difference
#   - the largest logprob difference over the agreeing tokens
# "ids" feeds llama.cpp's own prompt token ids (tests the forward pass alone);
# "text" feeds the prompt text (also tests our tokenizer).
use strict;
use warnings;
use Getopt::Long;
use File::Temp qw(tempfile);

my ($bin, $mode, $tol, $threads) = ('./hfcpu', 'both', 0.15, undef);
my $neartie;
GetOptions('bin=s' => \$bin, 'mode=s' => \$mode, 'tol=f' => \$tol, 'threads=i' => \$threads) or die "bad options\n";
my ($model, $golden) = @ARGV;
$neartie = $tol < 0.1 ? 0.1 : $tol;   # a divergence is acceptable when the golden top-2 gap is within the allowed noise
die "usage: $0 MODEL.gguf GOLDEN [--bin ./hfcpu] [--mode ids|text|both] [--tol 0.05]\n" unless $golden;

open(my $gf, '<', $golden) or die "cannot open $golden\n";
my @prompts; my $cur;
while (<$gf>) {
    chomp;
    if (/^prompt (\d+) (\S+)/) { $cur = { idx => $1, name => $2, lp => [] }; push @prompts, $cur; }
    elsif (/^text_hex (\S*)/ && $cur) { $cur->{text} = pack('H*', $1); }
    elsif (/^prompt_tokens (.*)$/ && $cur) { $cur->{ptoks} = $1; }
    elsif (/^gen_tokens (.*)$/ && $cur) { $cur->{gen} = [ split ' ', $1 ]; }
    elsif (/^logprob \d+ (\d+) (\S+) (.*)$/ && $cur) { $cur->{lp}[$1] = { lp => $2, top => $3 }; }
    elsif (/^n_predict (\d+)/) { $main::npred = $1; }
}
close $gf;
my $npred = $main::npred // 48;

my $bad = 0;
for my $p (@prompts) {
    for my $m (grep { $mode eq 'both' || $mode eq $_ } qw(ids text)) {
        my ($tf, $tpath) = tempfile(UNLINK => 1);
        my $cmd;
        if ($m eq 'ids') {
            $cmd = "'$bin' --stdin-mode none --op generate --model '$model' --prompt-ids '$p->{ptoks}'";
        } else {
            binmode $tf; print $tf $p->{text}; close $tf;
            $cmd = "'$bin' --stdin-mode none --op generate --model '$model' --prompt-file '$tpath'";
        }
        $cmd .= " --temp 0 --max-tokens $npred --logprobs 5";
        $cmd .= " --threads $threads" if $threads;
        my $out = `$cmd 2>&1`;
        my @tok = $out =~ /^\@token pos=(\d+) id=(\d+) logprob=(\S+) top="([^"]*)"/mg;
        my ($pref) = $out =~ /^\@prefill tokens=(\d+) ms=(\S+) tok_per_s=(\S+)/m;
        my ($gen_tps) = $out =~ /^\@gen .*tok_per_s=(\S+)/m;
        my ($prefill_tps) = $out =~ /^\@prefill .*tok_per_s=(\S+)/m;
        if ($out =~ /^\@done status=error.*msg=(.*)$/m) { printf "%-8s %-5s ERROR %s\n", $p->{name}, $m, $1; $bad++; next; }
        if ($out !~ /^\@done status=ok/m) {            # no response at all: crash, stale binary, wrong path ...
            my $first = join(" | ", grep { length } (split /\n/, $out)[0 .. 2]);
            $first = substr($first, 0, 300);
            printf "%-8s %-5s ERROR no valid response from %s (exit %d): %s\n", $p->{name}, $m, $bin, $? >> 8, $first eq '' ? '(no output)' : $first;
            $bad++; next;
        }
        my ($agree, $worst, $sum_abs, $sum_signed) = (0, 0, 0, 0);
        my $n = @tok / 4;
        for my $i (0 .. $n - 1) {
            my ($id, $lp) = ($tok[4 * $i + 1], $tok[4 * $i + 2]);
            last if !defined $p->{gen}[$i] || $id != $p->{gen}[$i];
            $agree++;
            my $d = $lp - $p->{lp}[$i]{lp};
            $sum_abs += abs($d); $sum_signed += $d;
            $worst = abs($d) if abs($d) > $worst;
        }
        my $total = scalar @{$p->{gen}};
        my $note = '';
        my $good = $agree > 0 && $worst <= $tol;
        if ($agree < $total && $agree < $n + 1 && $p->{lp}[$agree]) {   # diverged: was it a near-tie in the reference?
            my @top = map { (split /:/)[1] } split ' ', $p->{lp}[$agree]{top};
            my $gap = @top > 1 ? $top[0] - $top[1] : 99;
            if ($gap <= $neartie) { $note = sprintf(" (diverged at a near-tie, gap %.3f)", $gap); }
            else { $good = 0; $note = sprintf(" (diverged with gap %.3f > %.3f)", $gap, $neartie); }
        }
        $bad++ unless $good;
        printf "%-8s %-5s %s  agree %2d/%d tokens, max |dlogprob| %.4f, mean %.4f, bias %+.4f%s%s%s\n", $p->{name}, $m, $good ? 'OK  ' : 'FAIL', $agree, $total, $worst,
            $agree ? $sum_abs / $agree : 0, $agree ? $sum_signed / $agree : 0, $note,
            defined $prefill_tps ? sprintf("  prefill %.1f t/s", $prefill_tps) : '',
            defined $gen_tps ? sprintf("  decode %.1f t/s", $gen_tps) : '';
        if ($agree < $total && $agree < $n) {
            printf "           first difference at token %d: golden %s, ours %s\n", $agree, $p->{gen}[$agree] // '-', $tok[4 * $agree + 1] // '-';
        }
    }
}
print $bad ? "\n$bad check(s) failed\n" : "\nall checks passed\n";
exit($bad ? 1 : 0);
