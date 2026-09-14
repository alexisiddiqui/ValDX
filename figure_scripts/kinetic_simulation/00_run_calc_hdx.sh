#!/usr/bin/env bash
set -euo pipefail

WORK="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${WORK}/../.." && pwd)"
PACK="${ROOT}/figure_scripts/jaxent_autovalidation/_Bradshaw/Reproducibility_pack_v2"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="${WORK}/_output/calchdx_${STAMP}"
mkdir -p "${WORK}/_output"
mkdir "${RUN_DIR}"

cd "${RUN_DIR}"
export PYTHONPATH="${PACK}/code/calc_hdx${PYTHONPATH:+:${PYTHONPATH}}"
python "${PACK}/code/calc_hdx/calc_hdx.py" \
  -t "${PACK}/data/trajectories/TeaA_closed_reimaged.xtc" \
     "${PACK}/data/trajectories/TeaA_open_reimaged.xtc" \
  -p "${PACK}/data/trajectories/TeaA_ref_closed_state.pdb" \
  -m Radou -dt 0.167 1.0 10.0 60.0 120.0 \
  -mopt "{'save_detailed': True, 'contact_method': 'cutoff'}" \
  -seg "${PACK}/code/calc_hdx/data/TeaA_byresidue_segments.dat" -out TeaA_kin_

ln -sfn "calchdx_${STAMP}" "${WORK}/_output/calchdx_latest"
printf '%s\n' "${RUN_DIR}"
