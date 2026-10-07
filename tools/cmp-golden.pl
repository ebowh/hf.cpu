#!/usr/bin/perl
# cmp-golden.pl A.golden B.golden - how far apart are two llama.cpp recordings (e.g. repack on/off)? Measures the reference noise floor.
use strict; use warnings;
sub load { my ($f)=@_; open my $h,'<',$f or die; my ($cur,%p); while(<$h>){ chomp;
  if(/^prompt \d+ (\S+)/){$cur=$1} elsif(/^gen_tokens (.*)/){$p{$cur}{gen}=[split ' ',$1]}
  elsif(/^logprob \d+ (\d+) (\S+) (.*)/){ $p{$cur}{top}[$1]={ map{split /:/} split ' ',$3 } } } \%p }
my ($x,$y)=(load($ARGV[0]),load($ARGV[1]));
for my $n (sort keys %$x){ my @g=@{$x->{$n}{gen}}; my @h=@{$y->{$n}{gen}}; my $i=0; $i++ while $i<@g&&$i<@h&&$g[$i]==$h[$i];
  my ($mx,$sum,$c)=(0,0,0);
  for my $k (0..$i-1){ for my $id (keys %{$x->{$n}{top}[$k]}){ next unless exists $y->{$n}{top}[$k]{$id}; my $d=abs($x->{$n}{top}[$k]{$id}-$y->{$n}{top}[$k]{$id}); $mx=$d if $d>$mx; $sum+=$d;$c++ } }
  printf "%-8s agree %2d/%d  top5 max diff %.4f mean %.4f\n",$n,$i,scalar @g,$mx,$c?$sum/$c:0;
  if($n eq 'chatml'){ for my $f(0,1){ my $t=($x,$y)[$f]->{$n}{top}[26]; print "   ",join(' ',map{"$_:$t->{$_}"} sort{$t->{$b}<=>$t->{$a}} keys %$t),"\n" } } }
