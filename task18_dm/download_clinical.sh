#!/usr/bin/env bash
# Task 18 sensitivity analysis: body measures, blood pressure, smoking and diabetes questionnaire files (CDC/NCHS, public domain).
# Same layer folders as download_dm.sh; checks arrival and SHA256 only.
set -u
EXT="$(cd "$(dirname "$0")/.." && pwd)/external_data"
LAB=https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public
get() { local f="$1/$(basename "$2")"; [ -s "$f" ] && return 0; curl -sSfL --retry 3 -o "$f.part" "$2" && mv "$f.part" "$f" && echo "ok   $2" || { rm -f "$f.part"; echo "FAIL $2"; }; }
for spec in dev:C:2003:_C dev:D:2005:_D holdout:E:2007:_E holdout:F:2009:_F holdout:A:1999: holdout:B:2001:_B; do
  IFS=: read layer c y s <<< "$spec"; d="$EXT/nhanes_$layer/$c"
  for f in BMX BPX SMQ DIQ; do get "$d" "$LAB/$y/DataFiles/$f$s.xpt"; done
  ( cd "$d" && sha256sum *.xpt *.dat > SHA256SUMS.txt )
done
