#!/bin/bash

port=11726
ssh=root@ssh5.vast.ai

scp -P $port -r $ssh:~/MTAD-GAT/models ./

for i in {1..8}
do
    scp -P $port -r $ssh:~/MTAD-GAT/output/SMD/1-$i ./output/SMD/
done

for i in {1..9}
do
    scp -P $port -r $ssh:~/MTAD-GAT/output/SMD/2-$i ./output/SMD/
done

for i in {1..11}
do
    scp -P $port -r $ssh:~/MTAD-GAT/output/SMD/3-$i ./output/SMD/
done
