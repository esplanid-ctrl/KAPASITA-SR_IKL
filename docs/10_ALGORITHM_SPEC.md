# Algorithm Specification

## Inputs
Validated features at a common administrative level and period.

## Transform
Percentile-based need normalization, with direction determined by the indicator dictionary.

## Score
`SR_ICSS = 0.30*INC + 0.25*CAP + 0.20*SOC + 0.15*DIG + 0.10*POL`

## Confidence
Separate score:
`confidence = 0.4*completeness + 0.3*recency + 0.3*join_quality`

## Sensitivity
Run:
- baseline;
- INC +0.05 / SOC -0.05;
- CAP +0.05 / DIG -0.05.

Report rank stability. A score is not causal evidence.
