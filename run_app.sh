#!/usr/bin/env bash
# @date: 2018/7/13 14:22
# @author: wangke
# @concat: 891681500@qq.com
# =========================
model_dir=/workspace/model/
project_dir=/workspace/wide_deep_demo/
hdp_dir=/home/hdp_lbg_ectech/resultdata/strategy/deeplearning/wangke/wide_deep_demo/

mkdir -p ${model_dir}
mkdir -p ${project_dir}

hadoop fs -get ${hdp_dir}/* ${project_dir}

cd ${project_dir}
python wide_deep.py \
    --model_dir=${model_dir} \
    --ps_hosts="$PS" \
    --worker_hosts="$WORKER" \
    --job_name="$JOB_NAME" \
    --task_index="$TASK_ID"
