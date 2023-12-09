# Purpose and Results
This is the project that studies the effect of communication/information latency on the effectiveness of stop-and-go wave dissipation.
In particular, we study the effect of such latency in a ring of 22 vehicles, with one of them using a dissipation controller (the control car), while the 21 others use IDM
to imitate human behavior.
The control car uses three pieces of information as input: the ego car velocity, the relative velocity of the lead car, and the lead car distance.
Wrt the control car, we induce latency on the information itself, ranging from no latency (0 second delay) to 5 second delay (the three pieces of information are 5 seconds old).

The underlying framework we use to simulate this is Flow/Sumo, which is what the CIRCLES project has used historically to analyze stop-and-go dissipation.
Overall, we generally found what we expected to find - increased latency results in more inconsistent stop-and-go dissipation.

# Practical Overview
## How to Install prerequisites and jump into presumed operating environment
An unfortunate problem with Flow and Sumo is that it has a lot of dependency hell involved and code rot
due to being neglected for a few years. Thus, if you want to run the latest version of Flow and Sumo like we did, you should do these things in this order to bring yourself into a proper shell environment to invoke commands:

1. Run on an x64 ubuntu machine (we used 22.04).
2. Install singularity (basically docker with supercomputer security related augmentations): [Singularity Installation](https://docs.sylabs.io/guides/3.0/user-guide/installation.html#install-the-debian-ubuntu-package-using-apt)
3. Download a flow/sumo docker container using this command:
    singularity pull docker://fywu85/flow-desktop:latest
This generates a SIF file that we can use later.
4. Only problem with this container - the flow package is out of date. So we need to supply the latest version ourselves:
    git clone https://github.com/flow-project/flow.git
5. Now we can enter into the singularity shell itself - use the SIF file generated in step #3:
    singularity shell $FLOW_DESKTOP_CONTAINER_PATH
6. Now, export a new python path like so:
    export PYTHONPATH=$NEW_FLOW_GIT_FROM_STEP_4:$PATH_TO_FLOW_PROJECT_REPO:$PYTHONPATH
As you can see here, we are adding two folders, THIS github repo, and the new flow repo we downloaded.
7. Run this command:
    mkdir -p ~/ray_results/stabilizing_the_ring
We need this to store our tensorboard logs and checkpoints when you initially run the baseline experiments.
Now you should be ready to replicate our experiments - or at least the evaluation part!
## Commands to replicate evaluation
For the GUI examples, bear in mind that the first simulation gui that appears is a "false" gui - you have to hit play on that, watch it disappear, and then hit play on the next GUI that appears.
The 100 run examples will produce a numpy array printed out a la the results you see in grid_search_results.py.
### Replicating the "baseline" experiments - single run with GUI
Run this line:
    python3 test_baseline.py --num_rollouts 1 --render_mode sumo_gui
### Replicating the "baseline" experiments - 100 runs with no GUI
Run this line:
    python3 test_baseline.py --num_rollouts 100 --render_mode none
### Replicating the PI saturation experiments - single run with GUI
Run this line - replace $DELAY with your latency in seconds:
    python3 test_pisaturation.py --num_rollouts 1 --render_mode sumo_gui --delay $DELAY
### Replicating the PI saturation experiments - 100 runs with no GUI
Run this line - replace $DELAY with your latency in seconds:
    python3 test_pisaturation.py --num_rollouts 100 --render_mode none --delay $DELAY
### Replicating the RL experiments - single run with GUI
Run this line - replace $RAY_RESULT_FOLDER with the immediate folder within "~/ray_results/stabilizing_the_ring" containing your experiment:
    python3 test_rllib.py $RAY_RESULT_FOLDER 2500 --num_rollouts 1 --render_mode sumo_gui
### Replicating the RL experiments - 100 runs with GUI
Run this line - replace $RAY_RESULT_FOLDER with the full path to the immediate folder within "~/ray_results/stabilizing_the_ring" containing your experiment:
    python3 test_rllib.py $RAY_RESULT_FOLDER 2500 --num_rollouts 100 --render_mode none

## Commands to replicate RL training:
Run this line for each desired delay :
    bash train_bash.bash $DELAY
This will generate in ~/ray_results/stabilizing_the_ring_delay_$DELAY your $RAY_RESULT_FOLDER. By default it trains for 2500 epochs.
We also have a SLURM script that you can modify to run on a supercomputer if you desire, but it was for own internal consumption.