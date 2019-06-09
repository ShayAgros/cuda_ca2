#!/bin/bash


maxLoad=25782.95

highEnd=$(echo "scale=3; $maxLoad*2" | bc)
lowEnd=$(echo "scale=3; $maxLoad/10" | bc)


step=$(echo "scale=3; ($highEnd-$lowEnd)/10" | bc)
echo $lowEnd - $highEnd

for i in `seq 0 9`; do
	load=$(echo "scale=3; ${lowEnd} + ${i}*${step}" | bc)
	#echo load is $load
	output=$(./ex2 streams $load | tail -n2)

	throughput=$(echo "${output}" | grep throughput | cut -d' ' -f3)
	latency=$(echo "${output}" | grep latency | cut -d' ' -f4)

	echo ${load} ${throughput} ${latency}
done

