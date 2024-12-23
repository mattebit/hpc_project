rm parallel_parameter.sh.*
rm bench.sh.*

sub_job() {
    # Args:
    # 1: NUM_PROCESSES
    # 2: SELECT
    # 3: NUM_CPUS
    qsub -v NUM_PROCESS=$1 -l select=$2:ncpus=$3:mem=64gb -o bench_$1_$2_$3.log parallel_parameter.sh
}

sub_job 1 1 1
sub_job 2 2 1
sub_job 4 2 2
sub_job 4 2 4

watch "qstat | grep $USER"