#!/usr/bin/env bash
# Task 18: public NHANES laboratory, demographic and linked mortality files (CDC/NCHS, public domain).
# Development cycles go to external_data/nhanes_dev/<cycle>, held-out cycles to external_data/nhanes_holdout/<cycle>.
# Only checks that each file arrived and records its SHA256; reads no content.
set -u
EXT="$(cd "$(dirname "$0")/.." && pwd)/external_data"
LAB=https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public
MORT=https://ftp.cdc.gov/pub/Health_Statistics/NCHS/datalinkage/linked_mortality

get() {  # dir url
  mkdir -p "$1"
  local f="$1/$(basename "$2")"
  [ -s "$f" ] && return 0
  curl -sSfL --retry 3 -o "$f.part" "$2" && mv "$f.part" "$f" && echo "ok   $2" || { rm -f "$f.part"; echo "FAIL $2"; }
}

cycle() {  # layer name year mortyears files...
  local d="$EXT/nhanes_$1/$2" y=$3 m=$4; shift 4
  for f in "$@"; do get "$d" "$LAB/$y/DataFiles/$f.xpt"; done
  get "$d" "$MORT/NHANES_${m}_MORT_2019_PUBLIC.dat"
  ( cd "$d" && sha256sum *.xpt *.dat > SHA256SUMS.txt )
  echo "NHANES $m laboratory/demographic files from $LAB/$y/DataFiles/ and linked mortality file from $MORT/, downloaded $(date +%F) for Task 18 (layer: $1)." > "$d/SOURCE.txt"
}

cycle dev     C 2003 2003_2004 DEMO_C L40_C L25_C L10_C L13_C L11_C L16_C
cycle dev     D 2005 2005_2006 DEMO_D BIOPRO_D CBC_D GHB_D HDL_D TCHOL_D CRP_D ALB_CR_D
cycle holdout E 2007 2007_2008 DEMO_E BIOPRO_E CBC_E GHB_E HDL_E TCHOL_E CRP_E ALB_CR_E
cycle holdout F 2009 2009_2010 DEMO_F BIOPRO_F CBC_F GHB_F HDL_F TCHOL_F CRP_F ALB_CR_F
cycle holdout A 1999 1999_2000 DEMO LAB18 LAB25 LAB10 LAB13 LAB11 LAB16
cycle holdout B 2001 2001_2002 DEMO_B L40_B L25_B L10_B L13_B L11_B L16_B
