#!/bin/bash
#SBATCH --job-name=fast_slicing
#SBATCH --time=36:00:00
#SBATCH --mem=8G
#SBATCH --mail-type=all
#SBATCH --mail-user=matthew.holden@carleton.ca

#### load and unload modules you may need
module load python/3.10
source home/holden/envs/vasospasm/bin/activate
module load gcc/9.3.0 opencv python scipy-stack

#### execute code
python -u image_slicing.py /home/holden/projects/def-holden/fast_ultrasound_data /home/holden/scratch/fast_sliced 0.2

