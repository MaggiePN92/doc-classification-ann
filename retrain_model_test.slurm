#!/bin/bash
#SBATCH --job-name=in5550_assignment1_magnuspn_victoste_rerun_best
#SBATCH --account=ec30
#SBATCH --time=04:30:00
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=32G
#
# by default, request two cores (NumPy may just know how to take
# advantage of them; for larger computations, maybe use between
# six and ten; at some point, we will look at how to run on gpus
#
#SBATCH --cpus-per-task=2

# NB: this script should be run with "sbatch ...
# See https://www.uio.no/english/services/it/research/platforms/edu-research/help/fox/jobs/submitting.md

source ${HOME}/.bashrc

# sanity: exit on all errors and disallow unset environment variables
set -o errexit
set -o nounset

# the important bit: unload all current modules (just in case) and load only the necessary ones
module purge
module use -a /fp/projects01/ec30/software/easybuild/modules/all/
module load nlpl-nlptools/2022.01-foss-2021a-Python-3.9.5
module load nlpl-pytorch/1.11.0-foss-2021a-cuda-11.3.1-Python-3.9.5
module load nlpl-gensim/4.2.0-foss-2021a-Python-3.9.5

# print information (optional)
echo "submission directory: ${SUBMITDIR}"

# by default, pass on any remaining command-line options
python3 retrain_best_mod.py /fp/projects01/ec30/IN5550/obligatories/1/signal_20_obligatory1_train.tsv.gz /fp/projects01/ec30/magvic_large_files/output/trgram_pos_vec cpu
