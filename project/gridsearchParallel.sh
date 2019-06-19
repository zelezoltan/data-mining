#!/bin/bash

find_best_parameter () {
    echo -e "Dataset: $1"

    learning_rates=(0.1 0.05 0.01 0.001)
    max_leaves=(2 3 6 11)
    n_estimators=(100 200 500 1000)
    subsamples=(0.6 0.7 0.8 1.0)

    FILENAME=$1
    FILENAME=${FILENAME%.*}
    FILENAME=${FILENAME##*/}
    FILENAME="results/$FILENAME.txt"
    echo "Output file: $FILENAME"
    > $FILENAME

    best_learning_rate=${learning_rates[0]}
    best_max_leaves=${max_leaves[0]}
    best_n_estimators=${n_estimators[0]}
    best_subsample=${subsamples[0]}
    best_accuracy=0
    iteration=0

    for estimators in ${n_estimators[*]}
    do
        for learning_rate in ${learning_rates[*]}
        do
            for leaves in ${max_leaves[*]}
            do
                for subsample in ${subsamples[*]}
                do
                    ((iteration++))
                    accuracy=$(python3 main.py $1 $estimators $learning_rate $leaves $subsample)
                    echo -en "\rIteration $iteration/256, accuracy: $accuracy"
                    if (( $(echo "$accuracy > $best_accuracy" |bc -l) ))
                    then
                        best_accuracy=$accuracy
                        best_learning_rate=$learning_rate
                        best_max_leaves=$leaves
                        best_n_estimators=$estimators
                        best_subsample=$subsample
                        echo -e "\nNew best accuracy: $best_accuracy \n learning_rate: $learning_rate\n max_leaves: $leaves\n n_estimators: $estimators\n subsample: $subsample\n"
                        if (( $(echo "$best_accuracy == 1.0" |bc -l) ))
                        then
                            break 4
                        fi
                    fi
                done
            done
        done
    done
    ((i++))

    echo -e "\nBest accuracy: $best_accuracy\n"
    echo "Found best hyperparameters: " >> $FILENAME
    echo "learning_rate: $best_learning_rate" >> $FILENAME
    echo "max_leaf_nodes: $best_max_leaves" >> $FILENAME
    echo "n_estimators: $best_n_estimators" >> $FILENAME
    echo -e "subsample: $best_subsample\n" >> $FILENAME
    echo "Accuracy: $best_accuracy" >> $FILENAME
    echo "------------------------------------------------------------" >> $FILENAME
}

export -f find_best_parameter

if [ ! -d "results" ]
then
    mkdir "results"
fi

sudo apt install parallel -y
echo -e "\n____________________________________\nReading datasets...\n"
parallel -j 4 "find_best_parameter" ::: "data"/* 