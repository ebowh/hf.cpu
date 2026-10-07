#!/usr/bin/perl
# gguf-inspect.pl - independent GGUF reader in pure Perl (core modules only).
#
#   perl tools/gguf-inspect.pl [--events] [--tensors] [--kv] FILE.gguf
#
# Prints a human-readable summary, or with --events the same
# "@name key=value" lines that `hfcpu --op inspect` emits for the shared
# subset (@file, @types). The two readers are separate implementations, so
# comparing them (tests/e2e.pl) cross-checks both.
use strict;
use warnings;
use Getopt::Long;

my ($events, $tensors, $show_kv) = (0, 0, 0);
GetOptions('events' => \$events, 'tensors' => \$tensors, 'kv' => \$show_kv) or die "bad options\n";
my $path = shift or die "usage: $0 [--events] [--tensors] [--kv] FILE\n";

# id => [name, block elements, bytes per block]
my %TYPES = (
    0 => ['f32', 1, 4], 1 => ['f16', 1, 2], 2 => ['q4_0', 32, 18], 3 => ['q4_1', 32, 20],
    6 => ['q5_0', 32, 22], 7 => ['q5_1', 32, 24], 8 => ['q8_0', 32, 34], 9 => ['q8_1', 32, 36],
    10 => ['q2_K', 256, 84], 11 => ['q3_K', 256, 110], 12 => ['q4_K', 256, 144],
    13 => ['q5_K', 256, 176], 14 => ['q6_K', 256, 210], 15 => ['q8_K', 256, 292],
    16 => ['iq2_xxs', 256, 66], 17 => ['iq2_xs', 256, 74], 18 => ['iq3_xxs', 256, 98],
    19 => ['iq1_s', 256, 50], 20 => ['iq4_nl', 32, 18], 21 => ['iq3_s', 256, 110],
    22 => ['iq2_s', 256, 82], 23 => ['iq4_xs', 256, 136], 24 => ['i8', 1, 1], 25 => ['i16', 1, 2],
    26 => ['i32', 1, 4], 27 => ['i64', 1, 8], 28 => ['f64', 1, 8], 29 => ['iq1_m', 256, 56],
    30 => ['bf16', 1, 2], 34 => ['tq1_0', 256, 54], 35 => ['tq2_0', 256, 66], 39 => ['mxfp4', 32, 17],
    40 => ['nvfp4', 64, 36], 41 => ['q1_0', 128, 18], 42 => ['q2_0', 64, 18],
    142 => ['pq2_0', 128, 34], 143 => ['ptq1_0', 128, 28],
);
my @KVT = qw(u8 i8 u16 i16 u32 i32 f32 bool string array u64 i64 f64);

open(my $fh, '<:raw', $path) or die "cannot open $path: $!\n";
my $size = -s $fh;
local $/;
my $data = <$fh>;
close $fh;
my $pos = 0;

sub need { my $n = shift; die "truncated file at byte $pos\n" if $pos + $n > length $data; }
sub u32 { need(4); my $v = unpack('V', substr($data, $pos, 4)); $pos += 4; $v }
sub u64 { need(8); my $v = unpack('Q<', substr($data, $pos, 8)); $pos += 8; $v }
sub str {
    my $n = u64(); need($n);
    my $s = substr($data, $pos, $n); $pos += $n; $s
}
sub scalar_val {
    my $t = shift;
    my %fmt = (0 => ['C', 1], 1 => ['c', 1], 2 => ['v', 2], 3 => ['s<', 2], 4 => ['V', 4],
               5 => ['l<', 4], 6 => ['f<', 4], 7 => ['C', 1], 10 => ['Q<', 8], 11 => ['q<', 8], 12 => ['d<', 8]);
    my $f = $fmt{$t} or die "bad scalar type $t\n";
    need($f->[1]);
    my $v = unpack($f->[0], substr($data, $pos, $f->[1])); $pos += $f->[1];
    return $v;
}

die "not a GGUF file\n" unless substr($data, 0, 4) eq 'GGUF';
$pos = 4;
my $version = u32();
die "unsupported GGUF version $version\n" unless $version == 2 || $version == 3;
my $nt = u64();
my $nkv = u64();
my %kv;
my @kvorder;
my $align = 32;
for (1 .. $nkv) {
    my $k = str();
    my $t = u32();
    my ($val, $desc);
    if ($t == 8) { $val = str(); $desc = $val; }
    elsif ($t == 9) {
        my $et = u32(); my $cnt = u64();
        for (1 .. $cnt) { if ($et == 8) { str() } else { scalar_val($et) } }
        $val = undef; $desc = "array<$KVT[$et]>[$cnt]";
    } else { $val = scalar_val($t); $desc = $val; }
    $kv{$k} = $val;
    push @kvorder, [$k, $KVT[$t], $desc];
    $align = $val if $k eq 'general.alignment';
}
my @tens;
for (1 .. $nt) {
    my $name = str();
    my $nd = u32();
    my @ne = map { u64() } 1 .. $nd;
    my $type = u32();
    my $off = u64();
    my $nel = 1; $nel *= $_ for @ne;
    my $ti = $TYPES{$type};
    my $nbytes = 0;
    if ($ti) {
        die "tensor $name: row not a multiple of block size\n" if $ne[0] % $ti->[1];
        $nbytes = ($nel / $ti->[1]) * $ti->[2];
    }
    push @tens, { name => $name, ne => \@ne, type => $type, off => $off, nbytes => $nbytes };
}
my $data_off = int(($pos + $align - 1) / $align) * $align;
for my $t (@tens) {
    die "tensor $t->{name} extends past end of file\n" if $t->{nbytes} && $data_off + $t->{off} + $t->{nbytes} > $size;
}

my (%tn, %tb);
my $total = 0;
for my $t (@tens) { $tn{$t->{type}}++; $tb{$t->{type}} += $t->{nbytes}; $total += $t->{nbytes}; }

if ($events) {
    printf "\@file path=%s bytes=%d gguf_version=%d tensors=%d kv=%d\n", $path, $size, $version, $nt, $nkv;
    for my $id (sort { $a <=> $b } keys %tn) {
        my $name = $TYPES{$id} ? $TYPES{$id}[0] : "type_$id";
        printf "\@types name=%s id=%d tensors=%d bytes=%d\n", $name, $id, $tn{$id}, $tb{$id};
    }
    printf "\@summary tensor_bytes=%d\n", $total;
    exit 0;
}
printf "file:      %s (%d bytes)\nversion:   %d\ntensors:   %d\nmetadata:  %d keys\nalignment: %d\ndata at:   %d\n",
       $path, $size, $version, $nt, $nkv, $align, $data_off;
if ($show_kv) { printf "  %-48s %-8s %s\n", @$_ for map { [ $_->[0], $_->[1], length($_->[2]) > 80 ? substr($_->[2], 0, 80) . '...' : $_->[2] ] } @kvorder; }
printf "architecture: %s\n", $kv{'general.architecture'} // '(none)';
print "tensor types:\n";
for my $id (sort { $a <=> $b } keys %tn) {
    my $name = $TYPES{$id} ? $TYPES{$id}[0] : "type_$id (unknown)";
    printf "  %-14s %5d tensors %14d bytes\n", $name, $tn{$id}, $tb{$id};
}
printf "total tensor bytes: %d (%.2f MiB)\n", $total, $total / 1048576;
if ($tensors) {
    for my $t (@tens) {
        printf "  %-48s %-8s %-18s %d\n", $t->{name}, ($TYPES{$t->{type}} ? $TYPES{$t->{type}}[0] : "type_$t->{type}"),
               join('x', @{$t->{ne}}), $t->{nbytes};
    }
}
