#!/bin/sh
mkdir -p out
javac -d out src/vectordb/*.java && java -cp out vectordb.Main "$@"