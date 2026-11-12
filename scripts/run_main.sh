#!/usr/bin/env bash
# Tuned FedGRACE configurations of the main results (selected by validation NDCG@10, seed 42).
# Shared settings: lr_eta 80, latent_dim 32, cf_dim 64, K = 33, anchor_seed 42, 300 rounds, patience 50.
#
#   bash scripts/run_main.sh <backbone> <dataset> [extra main.py arguments, e.g. --seed 43 --device cuda:1]
#   bash scripts/run_main.sh all                    # every backbone x dataset, one after another
set -euo pipefail
cd "$(dirname "$0")/.."

config() {
  case "$1/$2" in
    fedncf/lastfm-2k)       echo "--lr 0.5 --lambda_orth 0.01 --lambda_align 0.1 --beta 1.0" ;;
    fedncf/amazon-office)   echo "--lr 0.5 --lambda_orth 0.01 --lambda_align 1.0 --beta 2.0" ;;
    fedncf/amazon-music)    echo "--lr 0.5 --lambda_orth 0.01 --lambda_align 0.1 --beta 2.0" ;;
    fedncf/amazon-science)  echo "--lr 0.5 --lambda_orth 5.0 --lambda_align 5.0 --beta 2.0" ;;
    fedncf/amazon-video)    echo "--lr 0.5 --lambda_orth 0.1 --lambda_align 0.1 --beta 0.5" ;;
    fedncf/amazon-toys)     echo "--lr 0.5 --lambda_orth 0.01 --lambda_align 1.0 --beta 0.5" ;;
    pfedrec/lastfm-2k)      echo "--lr 0.5 --lambda_orth 0.01 --lambda_align 0.1 --beta 1.0" ;;
    pfedrec/amazon-office)  echo "--lr 0.5 --lambda_orth 10.0 --lambda_align 10.0 --beta 4.0" ;;
    pfedrec/amazon-music)   echo "--lr 0.5 --lambda_orth 0.1 --lambda_align 5.0 --beta 2.0" ;;
    pfedrec/amazon-science) echo "--lr 0.5 --lambda_orth 5.0 --lambda_align 5.0 --beta 2.0" ;;
    pfedrec/amazon-video)   echo "--lr 0.5 --lambda_orth 10.0 --lambda_align 10.0 --beta 2.0" ;;
    pfedrec/amazon-toys)    echo "--lr 0.5 --lambda_orth 1.0 --lambda_align 5.0 --beta 0.25" ;;
    fedrap/lastfm-2k)       echo "--lr 1.0 --lr_personal 1.0 --lambda_orth 0.1 --lambda_align 0.1 --beta 1.0" ;;
    fedrap/amazon-office)   echo "--lr 4.0 --lr_personal 0.1 --lambda_orth 0.1 --lambda_align 0.1 --beta 2.0" ;;
    fedrap/amazon-music)    echo "--lr 2.0 --lr_personal 0.1 --lambda_orth 0.1 --lambda_align 0.1 --beta 2.0" ;;
    fedrap/amazon-science)  echo "--lr 2.0 --lr_personal 0.1 --lambda_orth 0.1 --lambda_align 0.1 --beta 2.0" ;;
    fedrap/amazon-video)    echo "--lr 4.0 --lr_personal 0.1 --lambda_orth 0.1 --lambda_align 0.1 --beta 1.0" ;;
    fedrap/amazon-toys)     echo "--lr 2.0 --lr_personal 0.1 --lambda_orth 0.1 --lambda_align 0.1 --beta 4.0" ;;
    *) echo "unknown backbone/dataset: $1/$2" >&2; return 1 ;;
  esac
}

run() {
  local backbone=$1 dataset=$2; shift 2
  # shellcheck disable=SC2046
  python main.py --backbone "$backbone" --dataset "$dataset" $(config "$backbone" "$dataset") \
    --lr_eta 80 --latent_dim 32 --cf_dim 64 --anchor_seed 42 --num_round 300 --early_stop_patience 50 "$@"
}

if [ "${1:-}" = all ]; then
  shift
  for backbone in fedncf pfedrec fedrap; do
    for dataset in lastfm-2k amazon-office amazon-music amazon-science amazon-video amazon-toys; do
      if [ ! -d "${FEDGRACE_DATASET_DIR:-dataset}/$dataset" ]; then
        echo "skipping $backbone/$dataset: no dataset folder" >&2; continue
      fi
      run "$backbone" "$dataset" "$@"
    done
  done
elif [ $# -ge 2 ]; then
  run "$@"
else
  sed -n '2,6p' "$0"; exit 1
fi
