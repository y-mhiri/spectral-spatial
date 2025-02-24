#/bin/sh

for mu in {0,1.0}
do
    for lmbda in {0.001,0.002,0.01}
    do
    for noise_level in {0.0001,0.001,0.01} 
    do
        echo -e "\t - options: \n\t\t '--mu': $mu \n\t\t '--lmbda': $lmbda \n\t\t '--noise_level': $noise_level"
    done
    done
done 