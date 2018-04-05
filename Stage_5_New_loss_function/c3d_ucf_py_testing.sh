
#$ -cwd -V
#$ -l h_rt=2:00:00
#$ -l coproc_p100=1
#$ -M cnlp@leeds.ac.uk
#$ -m be

echo "Starting c3d+ucf testing script..."

module load python
module load python-libs/3.1.0

python < 2.C3D_ucf_model_training_multigpu_testing.py >> 2.C3D_ucf_model_training_multigpu_py_testing.txt
