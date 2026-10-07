#!/usr/bin/perl
# golden.pl - record reference behaviour of a model from a running llama.cpp
# llama-server, as a small text file the engine's tests can compare against.
#
#   llama-server -m model.gguf --port 8080 -c 4096 -np 1 --threads 4
#   perl tools/golden.pl --out tests/golden/qwen2.5-0.5b-q4_k_m.golden [--server http://127.0.0.1:8080]
#
# Core modules only (HTTP::Tiny, JSON::PP). The script is tolerant of small
# differences between llama.cpp versions in the JSON field names, and reports
# exactly what it could not find.
use strict;
use warnings;
use Getopt::Long;
use HTTP::Tiny;
use JSON::PP;
use POSIX qw(strftime);

my ($server, $out, $npredict, $nprobs) = ('http://127.0.0.1:8080', undef, 48, 5);
GetOptions('server=s' => \$server, 'out=s' => \$out, 'n-predict=i' => \$npredict, 'n-probs=i' => \$nprobs)
    or die "bad options\n";
defined $out or die "usage: $0 --out FILE [--server URL] [--n-predict N] [--n-probs N]\n";

my $http = HTTP::Tiny->new(timeout => 600);
my $json = JSON::PP->new->utf8->canonical;

sub post {
    my ($path, $body) = @_;
    my $r = $http->post("$server$path", { headers => { 'Content-Type' => 'application/json' }, content => $json->encode($body) });
    die "POST $path failed: $r->{status} $r->{reason} $r->{content}\n" unless $r->{success};
    return $json->decode($r->{content});
}
sub get {
    my $path = shift;
    my $r = $http->get("$server$path");
    return $r->{success} ? $json->decode($r->{content}) : {};
}

my $long = join(' ', map { "Sentence number $_ describes a small experiment in which a researcher measures how quickly a cached prefix can be reused." } 1 .. 18);
my @prompts = (
    ['plain',   "The capital of France is"],
    ['code',    "def fibonacci(n):\n    "],
    ['chatml',  "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nSay hello in three languages.<|im_end|>\n<|im_start|>assistant\n"],
    ['unicode', "\xd0\x9f\xd1\x80\xd0\xb8\xd0\xb2\xd0\xb5\xd1\x82, \xd0\xba\xd0\xb0\xd0\xba \xd0\xb4\xd0\xb5\xd0\xbb\xd0\xb0? \xe4\xbd\xa0\xe5\xa5\xbd\xef\xbc\x8c\xe4\xb8\x96\xe7\x95\x8c\xef\xbc\x81 \xf0\x9f\x99\x82 caf\xc3\xa9 na\xc3\xafve"],
    ['long',    "$long\nIn summary, the researcher concluded that"],
);

my $props = get('/props');
my $build = $props->{build_info} // 'unknown';
my $model = $props->{model_path} // $props->{default_generation_settings}{model} // 'unknown';

open(my $fh, '>', $out) or die "cannot write $out: $!\n";
print $fh "# hfcpu golden file, recorded from llama.cpp\n";
print $fh "format 1\n";
print $fh "recorded ", strftime('%Y-%m-%dT%H:%M:%SZ', gmtime), "\n";
print $fh "llama_build $build\n";
print $fh "model_path $model\n";
print $fh "n_predict $npredict\nn_probs $nprobs\n";

my $warn = 0;
my $idx = 0;
for my $p (@prompts) {
    my ($name, $text) = @$p;
    my $tok = post('/tokenize', { content => $text, add_special => JSON::PP::false, parse_special => JSON::PP::true });
    my @ptoks = @{ $tok->{tokens} // die "no tokens in /tokenize response\n" };
    my $res = post('/completion', {
        prompt => \@ptoks, n_predict => $npredict, temperature => 0, top_k => 1, seed => 1,
        n_probs => $nprobs, cache_prompt => JSON::PP::false, return_tokens => JSON::PP::true,
        stream => JSON::PP::false,
    });
    my @cp = @{ $res->{completion_probabilities} // [] };
    my @gen = @{ $res->{tokens} // [] };
    if (!@gen && @cp) { @gen = map { $_->{id} } grep { defined $_->{id} } @cp; }
    if (!@gen) { warn "WARNING: no generated token ids for prompt '$name' (server too old for return_tokens?)\n"; $warn++; }
    print $fh "\nprompt $idx $name\n";
    print $fh "text_hex ", unpack('H*', $text), "\n";
    print $fh "prompt_tokens ", join(' ', @ptoks), "\n";
    print $fh "gen_tokens ", join(' ', @gen), "\n";
    print $fh "gen_text_hex ", unpack('H*', $res->{content} // ''), "\n";
    my $pos = 0;
    for my $e (@cp) {
        my $lp = $e->{logprob};
        $lp = log($e->{prob}) if !defined $lp && defined $e->{prob} && $e->{prob} > 0;
        my $top = $e->{top_logprobs} // $e->{top_probs} // [];
        my @t;
        for my $t (@$top) {
            my $l = $t->{logprob};
            $l = log($t->{prob}) if !defined $l && defined $t->{prob} && $t->{prob} > 0;
            push @t, sprintf('%d:%.6f', $t->{id} // -1, $l // 0);
        }
        printf $fh "logprob %d %d %.6f %s\n", $idx, $pos, $lp // 0, join(' ', @t);
        $pos++;
    }
    if (!@cp) { warn "WARNING: no completion_probabilities for prompt '$name'\n"; $warn++; }
    printf STDERR "prompt %-8s: %3d prompt tokens, %3d generated\n", $name, scalar @ptoks, scalar @gen;
    $idx++;
}
close $fh;
print STDERR "wrote $out", ($warn ? " with $warn warning(s)" : ''), "\n";
exit($warn ? 1 : 0);
