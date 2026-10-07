#!/usr/bin/perl
# Compare hfcpu's forward pass with the independent Perl oracle (tools/ref-forward.pl)
# on tiny random models of every supported layout.
#   perl tests/oracle.pl [path/to/hfcpu]
use strict;
use warnings;
use File::Temp qw(tempdir);
use Cwd qw(abs_path);
use File::Basename qw(dirname);

my $root = abs_path(dirname(__FILE__) . '/..');
my $bin = shift // "$root/hfcpu";
my $tmp = tempdir(CLEANUP => 1);
my ($pass, $fail) = (0, 0);
sub ok { my ($c, $n) = @_; if ($c) { $pass++ } else { $fail++; print STDERR "FAIL: $n\n" } }

# arch, weight type, prompt length, logprob tolerance
my @cases = (
    ['qwen2', 'f32', 70, 0.004], ['qwen3', 'f32', 70, 0.004], ['llama', 'f32', 70, 0.004],
    ['qwen2', 'f16', 20, 0.004], ['qwen2', 'q8_0', 70, 0.12], ['llama', 'q8_0', 33, 0.12],
    ['qwen3', 'q8_0', 150, 0.12],            # 150 tokens: three KV blocks
);
my $seed = 11;
for my $c (@cases) {
    my ($arch, $wt, $n, $tol) = @$c;
    my $g = "$tmp/m-$arch-$wt.gguf";
    my $deqf = "$tmp/m-$arch-$wt-deq.gguf";
    system("perl '$root/tools/mkmodel.pl' '$g' --arch $arch --wtype $wt --seed " . $seed++ . " --deq '$deqf' >/dev/null") == 0 or die;
    srand($seed);
    my $ids = join(' ', map { int(rand(300)) } 1 .. $n);
    my $steps = 5;
    my @ref = grep { /^gen / } split /\n/, `perl '$root/tools/ref-forward.pl' '$g' '$ids' --steps $steps`;
    my @got = grep { /^\@token / } split /\n/, `'$bin' --stdin-mode none --op generate --model '$g' --prompt-ids '$ids' --temp 0 --max-tokens $steps --logprobs 5 2>/dev/null`;
    my $label = "$arch/$wt/$n";
    ok(@ref == $steps && @got == $steps, "$label: step counts (ref " . scalar(@ref) . ", got " . scalar(@got) . ")");
    my ($worst, $same_ids, $same_top, $worst_c) = (0, 1, 1, 0);
    for my $i (0 .. $steps - 1) {
        next unless $ref[$i] && $got[$i];
        my ($rid, $rlp) = $ref[$i] =~ /^gen \d+ (\d+) (\S+) top:/;
        my ($gid, $glp) = $got[$i] =~ /id=(\d+) logprob=(\S+)/;
        $same_ids = 0 if $rid != $gid;
        my $d = abs($rlp - $glp); $worst = $d if $d > $worst;
        my ($rtop) = $ref[$i] =~ /top: (.*)$/; my ($gtop) = $got[$i] =~ /top="([^"]*)"/;
        my %gl = map { split /:/ } split ' ', $gtop;                  # our top-5 as id => logprob
        for my $e (split ' ', $rtop) {
            my ($id, $lp) = split /:/, $e;
            if (exists $gl{$id}) { my $dd = abs($gl{$id} - $lp); $worst_c = $dd if $dd > $worst_c; }
        }
    }
    ok($same_ids, "$label: greedy tokens identical");
    if ($wt eq 'q8_0') {          # exact weights, f32 activations: isolates weight handling from activation quantization
        my @dq = grep { /^\@token / } split /\n/, `'$bin' --stdin-mode none --op generate --model '$deqf' --prompt-ids '$ids' --temp 0 --max-tokens $steps --logprobs 5 2>/dev/null`;
        my $w2 = 0;
        for my $i (0 .. $steps - 1) { my ($rlp) = $ref[$i] =~ /^gen \d+ \d+ (\S+) top:/; my ($glp) = $dq[$i] =~ /logprob=(\S+)/; my $d = abs($rlp - $glp); $w2 = $d if $d > $w2; }
        ok($w2 < 0.004, sprintf("$label: q8_0 weights vs dequantized f32 copy (max diff %.5f)", $w2));
    }
    ok($worst < $tol, sprintf("$label: max logprob difference %.5f < %g", $worst, $tol));
    ok($worst_c < 3 * $tol, sprintf("$label: top-5 candidate logprobs agree (max diff %.5f < %g)", $worst_c, 3 * $tol));
    printf "  %-14s max |dlogprob| = %.5f, candidates %.5f\n", $label, $worst, $worst_c;
}
printf "oracle: %d passed, %d failed\n", $pass, $fail;
exit($fail ? 1 : 0);
