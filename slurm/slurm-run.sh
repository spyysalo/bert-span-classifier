#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH -p gpu
#SBATCH -t 06:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks-per-node=1
#SBATCH --account=Project_2001426
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err

OUTPUT_DIR="output/$SLURM_JOBID"

function on_exit {
    rm -rf "$OUTPUT_DIR"
    rm -f jobs/$SLURM_JOBID
}
trap on_exit EXIT

if [ "$#" -ne 6 ] && [ "$#" -ne 7 ]; then
    echo "Usage: $0 model data_dir seq_len batch_size learning_rate epochs [model_dir]"
    exit 1
fi

MODEL="$1"
DATA_DIR="$2"
MAX_SEQ_LENGTH="$3"
BATCH_SIZE="$4"
LEARNING_RATE="$5"
EPOCHS="$6"
if [ "$#" -eq 7 ]; then
    modelparam="--model_dir $7"
else
    modelparam=""
fi

VOCAB="$(dirname "$MODEL")/vocab.txt"
CONFIG="$(dirname "$MODEL")/bert_config.json"

if [[ $MODEL =~ "uncased" ]]; then
    caseparam="--do_lower_case"
elif [[ $MODEL =~ "multilingual" ]]; then
    caseparam="--do_lower_case"
else
    caseparam=""
fi

rm -rf "OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

rm -f latest.out latest.err
ln -s logs/$SLURM_JOBID.out latest.out
ln -s logs/$SLURM_JOBID.err latest.err

module purge
module load tensorflow/2.0.0
source venv/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "START $SLURM_JOBID: $(date)"

srun python3 train.py \
    --replace_span "[unused1]" \
    --bert_config_file "$CONFIG" \
    --init_checkpoint "$MODEL" \
    --vocab_file "$VOCAB" \
    --learning_rate $LEARNING_RATE \
    --max_seq_length $MAX_SEQ_LENGTH \
    --batch_size $BATCH_SIZE \
    --num_train_epochs $EPOCHS \
    --train_data "$DATA_DIR/train.tsv" \
    --dev_data "$DATA_DIR/dev.tsv" \
    $caseparam \
    $modelparam

result=$(egrep '^Final dev accuracy:' logs/${SLURM_JOB_ID}.out | perl -pe 's/.*accuracy: (\S+)\%.*/$1/')

echo -n 'TEST-RESULT'$'\t'
echo -n 'init_checkpoint'$'\t'"$MODEL"$'\t'
echo -n 'data_dir'$'\t'"$DATA_DIR"$'\t'
echo -n 'max_seq_length'$'\t'"$MAX_SEQ_LENGTH"$'\t'
echo -n 'train_batch_size'$'\t'"$BATCH_SIZE"$'\t'
echo -n 'learning_rate'$'\t'"$LEARNING_RATE"$'\t'
echo -n 'num_train_epochs'$'\t'"$EPOCHS"$'\t'
echo "$result"

seff $SLURM_JOBID

echo "END $SLURM_JOBID: $(date)"
