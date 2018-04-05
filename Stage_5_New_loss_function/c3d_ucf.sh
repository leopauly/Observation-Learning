#$ -cwd -V
#$ -l h_rt=24:00:00
#$ -l coproc_k80=2
#$ -M cnlp@leeds.ac.uk

module load cuda
module load singularity
singularity exec --bind /nobackup/:/nobackup  /nobackup/containers/deeplearn_1_gpu.img python < 2.C3D_ucf_model_training_multigpu_.py >> 2.C3D_ucf_model_training_multigpu_.txt
