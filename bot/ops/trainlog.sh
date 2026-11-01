#!/bin/zsh
# trainlog.sh IT: print the gae/epoch/ev lines of up5's train IT log from whichever host trained it
it=$1
cd ${0:A:h:h} || exit 1
host=$(grep -E "up5: iteration $it trained on" runs/fleet/fleet.log | tail -1 | sed -E 's/.*trained on ([a-z0-9]+) .*/\1/')
echo "it $it host $host"
pat='"gae_lambda"|"epoch"|"ev_value"|"saved"|"ent_coef"'
case $host in
  lab0) ssh -o ConnectTimeout=15 lab0 "grep -E '$pat' /mnt/sda/share/slzhang/vio_lab0/sts2/bot/fleet/jobs/train_up5${it}_unified.log" | cut -c1-260 ;;
  apollo) ~/.runai/bin/runai training exec slz-sts2-cpu -p xkzhang -- bash -c "grep -E '$pat' /scratch/xkzhang/slzhang_cvpr27/sts2/bot/fleet/jobs/train_up5${it}_unified.log" | cut -c1-260 ;;
  apollo2) ~/.runai/bin/runai training exec slz-sts2-cpu2 -p xkzhang -- bash -c "grep -E '$pat' /scratch/xkzhang/slzhang_cvpr27/pod2/sts2/bot/fleet/jobs/train_up5${it}_unified.log" | cut -c1-260 ;;
  apollo3) ~/.runai/bin/runai training exec slz-sts2-cpu3 -p xkzhang -- bash -c "grep -E '$pat' /scratch/xkzhang/slzhang_cvpr27/pod3/sts2/bot/fleet/jobs/train_up5${it}_unified.log" | cut -c1-260 ;;
  *) echo "unknown host '$host' (lab1-3 not wired in)";;
esac
