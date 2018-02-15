
#$ -cwd -V
#$ -l h_rt=24:00:00
#$ -l coproc_k80=2
#$ -M cnlp@leeds.ac.uk

echo "Starting final_p_cnn.sh training script..."

module load python
module load python-libs/3.1.0

python 2.C3D_ucf_model_training_multigpu_.py
