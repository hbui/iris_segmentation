#!/bin/bash

for i in $(seq $1)
do
	work_queue_worker localhost 1024&
done
