#!/usr/bin/env bash
# @date: 2018/7/13 15:04
# @author: wangke
# @concat: 891681500@qq.com
# =========================

rsync --port=1015 -qzrtopg --progress --delete 10.252.20.247::wide_deep_demo ./wide_deep_demo/

hadoop fs -rmr /home/hdp_lbg_ectech/resultdata/strategy/deeplearning/wangke/wide_deep_demo/
hadoop fs -put ./wide_deep_demo/ /home/hdp_lbg_ectech/resultdata/strategy/deeplearning/wangke/
#hadoop fs -ls /home/hdp_lbg_ectech/resultdata/strategy/deeplearning/wangke/