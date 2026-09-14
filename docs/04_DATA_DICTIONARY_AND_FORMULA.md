# Data Dictionary & Formula

## Core fields
- `region_code`: kode wilayah standar.
- `region_name`: nama wilayah standar.
- `period`: periode observasi.
- `inclusive_school_coverage`: coverage satuan pendidikan inklusif.
- `inclusive_infra_coverage`: coverage indikator sarpras inklusif.
- `special_needs_signal`: sinyal kebutuhan peserta didik berkebutuhan khusus.
- `teacher_equity_gap`: kebutuhan akibat ketidakmerataan guru.
- `gtk_capacity_gap`: kebutuhan kapasitas SDM.
- `poverty_rate`: P0.
- `internet_teaching_coverage`: coverage internet untuk pengajaran.
- `policy_signal`: sinyal kebijakan berbasis evidence.
- `confidence`: kualitas data/coverage/join.

## Normalization
Untuk indikator bad-is-high:
`need = 100 * percentile_rank(x)`

Untuk indikator good-is-high:
`need = 100 * (1 - percentile_rank(x))`

## Subscores
`INC = 0.60*inclusive_service_gap + 0.40*inclusive_infra_gap`

`CAP = 0.70*teacher_equity_gap + 0.30*gtk_capacity_gap`

`SOC = poverty vulnerability percentile`

`DIG = digital support need percentile`

`POL = evidence-coded policy implementation signal`

## Overall
`SR_ICSS = 0.30*INC + 0.25*CAP + 0.20*SOC + 0.15*DIG + 0.10*POL`

Interpretation bands (MVP only):
0–19 very low, 20–39 low, 40–59 moderate, 60–79 high, 80–100 very high.

Bands are descriptive, not statutory thresholds.
