pkill -f learner_main.py
pkill -f actor_main.py
pkill -f parameter_broadcaster.py

logdir=${1:-dbg_logdir}
taskstr=${2:-gym,MountainCarContinuous-v0}
#taskstr=${2:-gym,MountainCar-v0}
#taskstr=${2:-dm_control,cartpole_balance}
#taskstr=${2:-dm_control,walker_stand}
#taskstr=${2:-dm_control,walker_run}
modelstr=${3:-2048,2048,2048}
episodes=${4:-100000}
n_actors=${5:-4}
batch=${6:-256}
spi=${7:-16}
quantize=${8:-32}
quantize_communicate=${9:-32}
actor_update_period=${10:-${actor_update_period:-30}}
weight_compress=${11:0}

echo Running logdir=$logdir taskstr=$taskstr modelstr=$modelstr episodes=$episodes n_actors=$n_actors batch=$batch spi=$spi quantize=$quantize quantize_communicate=$quantize_communicate actor_update_period=$actor_update_period

mkdir -p $logdir
mkdir -p $logdir/logs_stdout

python learner_main.py --actor_update_period $actor_update_period --taskstr $taskstr --model_str $modelstr --num_episodes $episodes --quantize_communication $quantize_communicate --quantize $quantize --replay_table_max_times_sampled $spi --batch_size $batch --n_actors $n_actors --logpath ${logdir}/logfiles/ --weight_compress $weight_compress > ${logdir}/logs_stdout/learner_out 2>&1 &
master_pid=$! 

python parameter_broadcaster.py --actor_update_period $actor_update_period --taskstr $taskstr --model_str $modelstr --num_episodes $episodes --quantize_communication $quantize_communicate --quantize $quantize --replay_table_max_times_sampled $spi --batch_size $batch --n_actors $n_actors --logpath ${logdir}/logfiles/ --weight_compress $weight_compress > ${logdir}/logs_stdout/broadcaster_out 2>&1 &

for i in `seq 1 $n_actors`; do 
    python actor_main.py --actor_update_period $actor_update_period  --taskstr $taskstr  --model_str $modelstr --num_episodes $episodes  --n_actors $n_actors  --quantize_communication $quantize_communicate --quantize $quantize --replay_table_max_times_sampled $spi --batch_size $batch --actor_id $i --logpath ${logdir}/logfiles/ --weight_compress $weight_compress  > ${logdir}/logs_stdout/actorid=${i}_out 2>&1 & 
    echo $!
done

wait $master_pid

sleep 1
