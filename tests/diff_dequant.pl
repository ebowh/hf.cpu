#!/usr/bin/perl
# Differential test of src/ggtype.c against upstream ggml-quants.c.
#
#   perl tests/diff_dequant.pl REFDIR
#
# REFDIR must contain ggml-common.h and ggml-quants.c from llama.cpp
# (ggml/src/). Nothing from upstream is copied into this repository: the
# dequantizer functions are extracted at test time into a temp directory,
# compiled next to our code, and compared bit for bit on random blocks.
use strict;
use warnings;
use File::Temp qw(tempdir);
use File::Basename qw(dirname);
use Cwd qw(abs_path);

my $ref = shift or die "usage: $0 REFDIR\n";
my $root = abs_path(dirname(__FILE__) . "/..");
for my $f (qw(ggml-common.h ggml-quants.c)) { -r "$ref/$f" or die "missing $ref/$f\n"; }

# id, upstream suffix, block struct
my @types = (
    [2,'q4_0','block_q4_0'], [3,'q4_1','block_q4_1'], [6,'q5_0','block_q5_0'], [7,'q5_1','block_q5_1'],
    [8,'q8_0','block_q8_0'], [10,'q2_K','block_q2_K'], [11,'q3_K','block_q3_K'], [12,'q4_K','block_q4_K'],
    [13,'q5_K','block_q5_K'], [14,'q6_K','block_q6_K'], [20,'iq4_nl','block_iq4_nl'],
    [23,'iq4_xs','block_iq4_xs'], [41,'q1_0','block_q1_0'], [42,'q2_0','block_q2_0'],
);
# size-only checks for types we have no dequantizer for yet
my @size_only = (
    [9,'block_q8_1'], [15,'block_q8_K'], [16,'block_iq2_xxs'], [17,'block_iq2_xs'], [18,'block_iq3_xxs'],
    [19,'block_iq1_s'], [21,'block_iq3_s'], [22,'block_iq2_s'], [29,'block_iq1_m'], [34,'block_tq1_0'],
    [35,'block_tq2_0'], [39,'block_mxfp4'], [40,'block_nvfp4'],
);

open(my $qf, '<', "$ref/ggml-quants.c") or die; local $/; my $src = <$qf>; close $qf;
my $code = '';
if ($src =~ /(static inline void get_scale_min_k4\(.*?\n\})/s) { $code .= "$1\n"; } else { die "get_scale_min_k4 not found\n"; }
for my $t (@types) {
    my $n = $t->[1];
    if ($src =~ /^void (dequantize_row_$n)\((.*?)\n\}\n/ms) {
        my $body = "void oracle_$n($2\n}\n";
        $body =~ s/^void oracle_$n\(\s*//;           # keep it simple: re-emit header below
        $code .= "void oracle_$n(" . $body;
    } else { die "dequantize_row_$n not found upstream\n"; }
}

my $dir = tempdir(CLEANUP => 1);
open(my $o, '>', "$dir/oracle.c") or die;
print $o <<"HDR";
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include "ggtype.h"
#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#undef GGML_COMMON_DECL_C
#define GGML_COMMON_IMPL_C
#include "ggml-common.h"
#define GGML_RESTRICT restrict
#define GGML_FP16_TO_FP32(x) hfc_f16_to_f32(x)
HDR
print $o $code;
print $o "\nstatic uint64_t rs = 0x9e3779b97f4a7c15ull;\nstatic uint8_t rb(void){ rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (uint8_t)(rs >> 24); }\n";
print $o "static int same(const float *a, const float *b, size_t n){ size_t i; for(i=0;i<n;i++){ if(isnan(a[i]) && isnan(b[i])) continue; if(memcmp(&a[i],&b[i],4)!=0) return 0; } return 1; }\n";
print $o "int main(void){ int bad = 0; int trial;\n";
for my $t (@types) {
    my ($id, $n, $bs) = @$t;
    print $o <<"T";
  { const hfc_type_info *ti = hfc_type_lookup($id);
    if (!ti) { printf("type $id missing\\n"); bad++; }
    else {
      if (ti->bytes != sizeof($bs)) { printf("size mismatch $n: ours %u upstream %zu\\n", ti->bytes, sizeof($bs)); bad++; }
      for (trial = 0; trial < 300; trial++) {
        enum { NB = 4 };
        $bs blocks[NB]; float a[NB * 256], b[NB * 256]; size_t i, nel = (size_t)ti->blck * NB;
        for (i = 0; i < sizeof blocks; i++) ((uint8_t *)blocks)[i] = rb();
        oracle_$n(blocks, a, (int64_t)nel);
        if (hfc_dequant_row($id, blocks, b, nel) != HFC_OK) { printf("dequant failed $n\\n"); bad++; break; }
        if (!same(a, b, nel)) { printf("MISMATCH $n trial %d\\n", trial); bad++; break; }
      }
      printf("%-8s ok bytes=%u blck=%u\\n", "$n", ti->bytes, ti->blck);
    } }
T
}
for my $t (@size_only) {
    my ($id, $bs) = @$t;
    print $o "  { const hfc_type_info *ti = hfc_type_lookup($id); if (!ti || ti->bytes != sizeof($bs)) { printf(\"size mismatch for type $id ($bs)\\n\"); bad++; } }\n";
}
print $o "  printf(\"%s\\n\", bad ? \"DIFF TEST FAILED\" : \"DIFF TEST PASSED\"); return bad != 0; }\n";
close $o;

my @cmd = ('gcc', '-std=gnu99', '-O1', '-w', "-I$root/src", "-I$ref", '-o', "$dir/oracle",
           "$dir/oracle.c", "$root/src/ggtype.c", "$root/src/mem.c", '-lm');
system(@cmd) == 0 or die "compile failed: @cmd\n";
exit(system("$dir/oracle") == 0 ? 0 : 1);
