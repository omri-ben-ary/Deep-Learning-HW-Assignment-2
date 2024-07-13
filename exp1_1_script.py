import subprocess

# Define the range of values for the parameters
L_values = [2, 4, 8, 16]
K_values = [32, 64]
# P_values = [2, 3]
# H_values = [100, 200]

# Iterate through all combinations of parameters
for K in K_values:
    for L in L_values:
        # for P in P_values:
            # for H in H_values:
                # Construct the command
        command = [
            "srun",
            "-c", "2",
            "--gres=gpu:1",
            "--pty",
            "python",
            "-m", "hw2.experiments",
            "run-exp",
            "-n", "exp1_1_L" + str(L) + "_K" + str(K),
            "-K", str(K),
            "-L", str(L),
            "-P", "2",
            "-H", "100",
            "--early-stopping", "8",
            "-M", "cnn",
            "-s", "42",
            "--epochs", "10"
        ]
        
        # Execute the command
        print("Running command: {}".format(' '.join(command)))
        subprocess.call(command)