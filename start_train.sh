export PYTHONPATH=`pwd`
cd /content/drive/MyDrive/Experiment/20210816/code/pointer_summarizer/
python training_ptr_gen/train.py \
`if [ $1 ]; then echo "-m $1"; fi`
