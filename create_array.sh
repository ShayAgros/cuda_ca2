#!/bin/bash

echo -n "load=["
cat $1 | tail -n +2 | cut -d' ' -f1 | sort -n | while read line; do
	echo -n "${line},"
done
echo "]"

echo -n "throughput=["
cat $1 | tail -n +2 | cut -d' ' -f2 | sort -n | while read line; do
	echo -n "${line},"
done
echo "]"

echo -n "latency=["
cat $1 | tail -n +2 | cut -d' ' -f3 | sort -n | while read line; do
	echo -n "${line},"
done
echo "]"
