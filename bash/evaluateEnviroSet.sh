#!/bin/bash

# 设置cpu内核数量

cpu_cores=$(nproc)
echo "cpu cores: $cpu_cores"

# 设置cpu频率
cpu_freq=$(lscpu | grep "CPU MHz" | awk '{print $3}')

for((i=0;i<=cpu_cores-1;i++));
do
  sudo cpufreq-set -c $i -g performance
  sudo cpufreq-set -c $i -f $cpu_freq
done