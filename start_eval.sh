cd /content/drive/MyDrive/Experiment/20201213/code/pointer_summarizer/
export PYTHONPATH=`pwd`
MODEL_PATH=$1
MODEL_NAME=$(basename $MODEL_PATH)
python training_ptr_gen/eval.py $MODEL_PATH

