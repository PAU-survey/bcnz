#!/usr/bin/env bash

root="/home/marberi/pau/photoz/bpz"
tar="bcnz_v1.4.0.tar"
bdir="/tmp/bcnz_v1.4.0"
rm -r $bdir

mkdir $bdir
mkdir $bdir/bin
mkdir $bdir/lib
mkdir $bdir/config
mkdir $bdir/descr
mkdir $bdir/modes
mkdir $bdir/priors
cp -r bin/*py $bdir/bin/.
cp -r bin/*sh $bdir/bin/.
cp -r lib/*py $bdir/lib/.
cp -r modes/*py $bdir/modes/.
cp -r config/*py $bdir/config/.
cp -r descr/*py $bdir/descr/.
cp -r priors/*py $bdir/priors/.
cp -r $root/FILTER $bdir/FILTER
cp -r $root/SED $bdir/SED
#cp -r /Users/marberi/data/tut_photoz $bdir/photoz
cp -r $root/spectras.txt $bdir/spectras.txt
mkdir $bdir/AB

cd "/tmp"
tar -cf $tar bcnz_v1.4.0
