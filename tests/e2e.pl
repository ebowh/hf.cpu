#!/usr/bin/perl
# End-to-end tests for the hfcpu binary: protocol modes, error handling, and a
# cross-check of the C GGUF reader against the independent Perl reader.
#   perl tests/e2e.pl [path/to/hfcpu]
use strict;
use warnings;
use File::Temp qw(tempdir);
use Cwd qw(abs_path);
use File::Basename qw(dirname);

my $root = abs_path(dirname(__FILE__) . '/..');
my $bin = shift // "$root/hfcpu";
-x $bin or die "no binary at $bin (run make)\n";
my $tmp = tempdir(CLEANUP => 1);
my ($pass, $fail) = (0, 0);
my $RS = "\x1e";
my $US = "\x1f";

sub ok { my ($c, $n) = @_; if ($c) { $pass++ } else { $fail++; print STDERR "FAIL: $n\n" } }

sub slurp { my $f = shift; open(my $h, '<:raw', $f) or return ''; local $/; my $d = <$h>; close $h; defined $d ? $d : '' }
sub spew { my ($f, $d) = @_; open(my $h, '>:raw', $f) or die; print $h $d; close $h }

# run($args_string, $stdin) -> ($stdout, $stderr, $exit_code)
sub run {
    my ($args, $in) = @_;
    spew("$tmp/in", $in // '');
    my $rc = system("'$bin' $args < '$tmp/in' > '$tmp/out' 2> '$tmp/err'");
    return (slurp("$tmp/out"), slurp("$tmp/err"), $rc >> 8 | ($rc & 127 ? 128 + ($rc & 127) : 0));
}
sub count { my ($s, $re) = @_; my $n = () = $s =~ /$re/g; $n }

# ---- basics ---------------------------------------------------------------
my ($o, $e, $x) = run('--version');
ok($x == 0 && $o =~ /^hfcpu /, 'version');
($o, $e, $x) = run('--help');
ok($x == 0 && $o =~ /--stdin-mode/, 'help lists options');
($o, $e, $x) = run('--bogus');
ok($x == 2 && $e =~ /unknown option/, 'unknown startup option exits 2');
($o, $e, $x) = run('--n 0');
ok($x == 2 && $e =~ /out of range/, 'range error exits 2');

# ---- record (prompts) mode ------------------------------------------------------
($o, $e, $x) = run('--op echo --system SYS --id r', "alpha${RS}beta${RS}");
ok($x == 0, 'records: exit 0');
my @resp = split(/\x1e/, $o);
ok(@resp == 2 && !grep({ !/^\@begin .*\@done status=ok/s } @resp), 'records: two requests');
ok(count($o, qr/\x1e/) == 2, 'records: each response ends with the separator byte');
ok($o =~ /\@prompt len=5\nalpha\n/ && $o =~ /\@prompt len=4\nbeta\n/, 'records: prompts echoed exactly');
ok($o =~ /\@system len=3\nSYS\n/, 'records: system prompt from command line');

($o, $e, $x) = run('--op echo', 'no separator at all');
ok(count($o, qr/^\@begin /m) == 1 && $o =~ /\@prompt len=18\n/, 'records: EOF terminates the last record');
($o, $e, $x) = run('--op echo', '');
ok($o eq '' && $x == 0, 'records: empty stdin gives no requests');
($o, $e, $x) = run('--op echo', "${RS}${RS}");
ok(count($o, qr/^\@done status=ok/m) == 2 && $o =~ /\@prompt len=0\n/, 'records: empty prompts are requests');
($o, $e, $x) = run('--op echo --record-sep LF', "one\ntwo\n");
ok(count($o, qr/^\@begin /m) == 2, 'records: custom separator');

# per-record header
($o, $e, $x) = run('--op echo', "--n 3 --id H1${US}prompt text${RS}after${RS}");
ok($o =~ /\@opt key=n value=3/ && $o =~ /\@begin id=H1/ && $o =~ /\@prompt len=11\nprompt text\n/, 'header overrides apply');
ok($o =~ /\@opt key=n value=1/, 'header does not leak into the next record');
($o, $e, $x) = run('--op echo', "--bogus${US}x${RS}ok${RS}");
ok($o =~ /code=EINVAL/ && count($o, qr/^\@done status=ok/m) == 1, 'bad header: error, stream stays in sync');
($o, $e, $x) = run('--op echo', "--cache-dir /x${US}x${RS}");
ok($o =~ /code=EINVAL.*startup/, 'session-only option rejected in a request');
($o, $e, $x) = run('--op echo --prompt P', "body${RS}");
ok($o =~ /code=EINVAL.*more than once/, 'prompt given twice is an error');

# oversize record is skipped and the stream stays in sync
($o, $e, $x) = run('--op echo --max-record 10', ('x' x 100) . "${RS}short${RS}");
ok($o =~ /code=ERANGE/ && $o =~ /\@prompt len=5\nshort\n/, 'oversize record rejected, next record served');
ok(count($o, qr/\x1e/) == 2, 'oversize: still one terminator per input record');

# large record round trip
my $big = join('', map { chr(32 + ($_ * 7) % 90) } 1 .. 3_000_000);
($o, $e, $x) = run('--op echo', $big . $RS);
ok(index($o, "\@prompt len=3000000\n$big\n") >= 0, '3 MB record round trip');

# prompt containing newlines and quotes is carried verbatim
($o, $e, $x) = run('--op echo', "line1\nline \"2\"\t\\x${RS}");
ok(index($o, "\@prompt len=17\nline1\nline \"2\"\t\\x\n") >= 0, 'prompt bytes preserved');

# ---- args mode ----------------------------------------------------------------------
($o, $e, $x) = run('--stdin-mode args --op echo --system S',
    "--prompt hi --n 2 --id a1\n\n# comment\n!stats\n--bogus\n--prompt 'two words' --id a2\n!quit\n--prompt never --id a3\n");
ok(count($o, qr/^\@begin /m) == 3, 'args: three requests then quit (including the failed one)');
ok($o =~ /\@begin id=a1/ && $o =~ /\@begin id=a2/ && $o !~ /a3/, 'args: !quit stops processing');
ok($o =~ /\@prompt len=9\ntwo words\n/, 'args: shell quoting');
ok($o =~ /^\@stats requests=1 errors=0/m, 'args: !stats');
ok($o =~ /code=EINVAL/, 'args: bad line reported');
($o, $e, $x) = run('--stdin-mode args --op echo', "--prompt-file $tmp/nonexistent --id f\n--op doctor --cache-dir /x\n");
ok($o =~ /code=EIO.*cannot read --prompt-file/ && $o =~ /code=EINVAL/, 'args: file and scope errors');

# ---- single-shot mode ------------------------------------------------------------------
spew("$tmp/p.txt", "from a file\n");
($o, $e, $x) = run("--stdin-mode none --op echo --system-file $tmp/p.txt --prompt X");
ok($x == 0 && $o =~ /\@system len=12\nfrom a file\n\n/ && $o =~ /\@prompt len=1\nX\n/, 'none: files and exit 0');
($o, $e, $x) = run('--stdin-mode none --op generate --prompt hi');
ok($x == 1 && $o =~ /code=ENOTSUP/, 'none: generate is a clean ENOTSUP, exit 1');
($o, $e, $x) = run('--stdin-mode none --op nosuch');
ok($x == 1 && $o =~ /unknown --op/, 'none: unknown op');
spew("$tmp/set.conf", "system = from settings\nn = 5\n");
($o, $e, $x) = run("--settings $tmp/set.conf --stdin-mode none --op echo --n 7");
ok($o =~ /\@opt key=system value="from settings"/ && $o =~ /\@opt key=n value=7/, 'settings file layered under the command line');

# ---- inspect + cross-check with the Perl reader ----------------------------------------
my $gg = "$tmp/syn.gguf";
system("perl '$root/tools/mkgguf.pl' '$gg' --layers 4 --seed 7 > /dev/null") == 0 or die "mkgguf failed";
($o, $e, $x) = run("--stdin-mode none --op inspect --model '$gg'");
ok($x == 0 && $o =~ /\@arch name=qwen2/, 'inspect: synthetic model');
my $perl = `perl '$root/tools/gguf-inspect.pl' --events '$gg'`;
my @c_lines = grep { /^\@(file|types|summary) / } split /\n/, $o;
my @p_lines = split /\n/, $perl;
sub norm { my $l = shift; $l =~ s/ path=\S+//; $l =~ s/ all_types_supported=\S+//; $l =~ s/ reference_dequant=\S+//; $l }
ok(join("\n", map { norm($_) } @c_lines) eq join("\n", map { norm($_) } @p_lines), 'inspect: C reader and Perl reader agree');
($o, $e, $x) = run("--stdin-mode none --op inspect --model '$gg' --list-tensors");
ok(count($o, qr/^\@tensor /m) == 4 * 8 + 2, 'inspect --list-tensors lists every tensor');
spew("$tmp/bad.gguf", join('', map { chr(int(rand(256))) } 1 .. 5000));
($o, $e, $x) = run("--stdin-mode none --op inspect --model '$tmp/bad.gguf'");
ok($x == 1 && $o =~ /code=EFORMAT/, 'inspect: garbage file rejected cleanly');
spew("$tmp/trunc.gguf", substr(slurp($gg), 0, 3000));
($o, $e, $x) = run("--stdin-mode none --op inspect --model '$tmp/trunc.gguf'");
ok($x == 1 && $o =~ /code=EFORMAT/, 'inspect: truncated file rejected cleanly');
($o, $e, $x) = run("--stdin-mode none --op inspect --model '$tmp/missing.gguf'");
ok($x == 1 && $o =~ /code=EIO/, 'inspect: missing file');
($o, $e, $x) = run('--stdin-mode none --op inspect');
ok($x == 1 && $o =~ /needs --model/, 'inspect: needs --model');

# ---- tokenize -----------------------------------------------------------------------------
my $vocab = "$root/tests/data/ggml-vocab-qwen2.gguf";
($o, $e, $x) = run("--stdin-mode none --op tokenize --model '$vocab' --prompt 'Hello world'");
ok($x == 0 && $o =~ /\@tokens count=2 ids="9707 1879"/, 'tokenize: Hello world');
($o, $e, $x) = run("--op tokenize --model '$vocab'", "<|im_start|>user\nhi<|im_end|>${RS}12345 \xc3\xa9${RS}");
ok($o =~ /ids="151644 872 198 6023 151645"/ && $o =~ /ids="16 17 18 19 20 3958"/, 'tokenize: special tokens and digits over stdin records');
($o, $e, $x) = run("--stdin-mode none --op tokenize --model '$vocab' --no-parse-special --prompt '<|im_end|>'");
ok($o =~ /count=([0-9]+)/ && $1 > 1 && $o !~ /151645/, 'tokenize: --no-parse-special treats them as text');
($o, $e, $x) = run("--stdin-mode none --op tokenize --model '$gg' --prompt hi");
ok($x == 1 && $o =~ /code=EFORMAT/, 'tokenize: model without tokenizer metadata is an error');
($o, $e, $x) = run('--stdin-mode none --op tokenize --prompt hi');
ok($x == 1 && $o =~ /needs --model/, 'tokenize: needs --model');

# ---- doctor ------------------------------------------------------------------------------
($o, $e, $x) = run("--stdin-mode none --op doctor --cache-dir $tmp/cache");
ok($x == 0 && $o =~ /\@profile source=measured/ && $o =~ /key=bw.best/ && $o =~ /\@mem total=/, 'doctor: measures and reports');
my @profs = glob("$tmp/cache/machine-*.prof");
ok(@profs == 1, 'doctor: profile written');
($o, $e, $x) = run("--stdin-mode none --op doctor --cache-dir $tmp/cache");
ok($o =~ /\@profile source=cache/, 'doctor: second run uses the cached profile');
($o, $e, $x) = run("--stdin-mode none --op doctor --cache-dir $tmp/cache --probe-force");
ok($o =~ /\@profile source=measured/, 'doctor: --probe-force re-measures');

# ---- generate: protocol, resident model, allocation failures ----------------------------------
my $tiny = "$tmp/tiny.gguf";
system("perl '$root/tools/mkmodel.pl' '$tiny' --arch qwen2 --wtype q8_0 --seed 4 >/dev/null") == 0 or die;
($o, $e, $x) = run("--stdin-mode none --op generate --model '$tiny' --prompt-ids '1 2 3 4 5' --temp 0 --max-tokens 6 --id g1");
ok($x == 0 && $o =~ /\@prefill tokens=5/ && $o =~ /\@gen tokens=6 .*stop=length/ && $o =~ /\@done status=ok/, 'generate: prefill, decode and stats events');
ok(count($o, qr/^\@token /m) == 6, 'generate: one  event per generated token');
($o, $e, $x) = run("--stdin-mode args --model '$tiny' --temp 0 --max-tokens 3", "--op generate --prompt-ids '1 2 3' --id a\n--op generate --prompt-ids '1 2 3' --id b\n");
my @gens = $o =~ /\@gen tokens=3 ms=\S+ tok_per_s=\S+ stop=length/g;
ok(@gens == 2, 'generate: two requests in one process reuse the resident model');
my @blocks = split /^\@begin /m, $o;
my ($t1) = ($blocks[1] // '') =~ /\@token pos=0 id=(\d+)/; my ($t2) = ($blocks[2] // '') =~ /\@token pos=0 id=(\d+)/;
ok(defined $t1 && defined $t2 && $t1 == $t2, 'generate: same prompt, same seed, same token');
($o, $e, $x) = run("--stdin-mode none --op generate --model '$tiny' --prompt hi");
ok($x == 1 && $o =~ /no usable tokenizer/, 'generate: text prompt without tokenizer is a clean error');
($o, $e, $x) = run("--stdin-mode none --op generate --model '$tiny' --prompt-ids '1 9999'");
ok($x == 1 && $o =~ /bad token id/, 'generate: token id out of range');
($o, $e, $x) = run("--stdin-mode none --op generate --model '$tiny' --prompt-ids '1 2' --system hello");
ok($x == 1 && $o =~ /chat templates are not implemented/, 'generate: --system explains the missing template support');
($o, $e, $x) = run("--stdin-mode none --op generate --model '$tiny' --prompt-ids '1 2 3 4 5 6 7 8 9' --ctx-max 8");
ok($x == 1 && $o =~ /--ctx-max/, 'generate: prompt longer than --ctx-max');
for my $n (1 .. 40) {
    my $r = system("HFC_FAIL_AFTER=$n '$bin' --stdin-mode none --op generate --model '$tiny' --prompt-ids '1 2 3 4 5' --temp 0 --max-tokens 3 > '$tmp/o3' 2> '$tmp/e3'");
    my $out = slurp("$tmp/o3");
    ok(($r & 127) == 0 && $out =~ /\@done status=(ok|error)/, "generate: allocation failure #$n handled (exit " . ($r >> 8) . ")");
}

# ---- threads: results must not depend on the thread count -----------------------------------
{
    my $ids = join(' ', map { ($_ * 37) % 300 } 1 .. 90);
    my @sums;
    for my $t (1, 2, 3, 4) {
        ($o, $e, $x) = run("--stdin-mode none --op generate --model '$tiny' --prompt-ids '$ids' --temp 0 --max-tokens 8 --logprobs 3 --threads $t");
        my @tl = grep { /^\@token / } split /\n/, $o;
        push @sums, join("\n", @tl);
        ok(@tl == 8 && $o =~ /threads=$t\b/, "threads=$t: ran with that many threads");
    }
    ok($sums[0] eq $sums[1] && $sums[1] eq $sums[2] && $sums[2] eq $sums[3], 'output is bit-identical for 1, 2, 3 and 4 threads');
    ($o, $e, $x) = run("--stdin-mode none --op bench --model '$tiny' --bench-prompt 64 --bench-gen 8 --bench-threads 1,2");
    my @b = $o =~ /\@bench threads=(\d+) prefill_tok_s=([0-9.]+) decode_tok_s=([0-9.]+)/g;
    ok(@b == 6 && $b[1] > 0 && $b[2] > 0 && $b[0] == 1 && $b[3] == 2, 'bench: one line per thread count with positive rates');
}

# ---- robustness: closed stdout must not crash or hang --------------------------------------
my $many = join('', map { "p$_$RS" } 1 .. 2000);
spew("$tmp/many", $many);
my $rc = system("'$bin' --op echo < '$tmp/many' | head -c 100 > /dev/null");
ok(($rc & 127) == 0, 'closed stdout: no signal death in the pipeline');

# ---- fault injection through the environment ------------------------------------------------
for my $n (1, 2, 3, 5, 8, 13, 21, 34) {
    my $r = system("HFC_FAIL_AFTER=$n '$bin' --stdin-mode none --op echo --prompt hi --system s > '$tmp/o2' 2> '$tmp/e2'");
    my $sig = $r & 127;
    ok($sig == 0, "alloc failure #$n: no crash (exit " . ($r >> 8) . ")");
}

printf "e2e: %d passed, %d failed\n", $pass, $fail;
exit($fail ? 1 : 0);
