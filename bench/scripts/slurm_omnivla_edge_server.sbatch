#!/usr/bin/env bash
#SBATCH --job-name=rvln-omnivla-server
#SBATCH --partition=main
#SBATCH --nodelist=pve1ubuntu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:07:00
#SBATCH --output=/mnt/workspace/nop/rvln/bench/runs/validation/omnivla_edge_server-%j.log

exec bash /mnt/workspace/nop/rvln/bench/scripts/run_omnivla_edge_server.sh
