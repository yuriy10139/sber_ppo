dir=`ls -la ./runs/ | grep Env | tail -n 1 | awk '{print $9}'`
tensorboard --logdir "./runs/$dir"
