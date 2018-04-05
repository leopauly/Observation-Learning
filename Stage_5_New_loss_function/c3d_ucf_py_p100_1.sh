
#$ -cwd -V
#$ -l h_rt=48:00:00
#$ -l coproc_p100=4
#$ -M cnlp@leeds.ac.uk
#$ -m be

echo "Starting c3d+ucf training script..."

module load python
module load python-libs/3.1.0

python < 2.C3D_ucf_model_training_multigpu_.py >> 2.C3D_ucf_model_training_multigpu_py_p100_1.txt
