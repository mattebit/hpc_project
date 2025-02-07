rm parallel.sh.*

qsub parallel.sh

watch "qstat | grep $USER"