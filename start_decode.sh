cd /content/drive/MyDrive/Experiment/20210816/code/pointer_summarizer/
export PYTHONPATH=`pwd`
MODEL=$1
python training_ptr_gen/decode.py $MODEL

