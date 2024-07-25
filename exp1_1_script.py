import subprocess

# #exp1_1
# # Define the range of values for the parameters
# L_values = [2, 4, 8, 16]
# K_values = [32, 64]
# # P_values = range(1,9)
# # H_values = [32]

# # Iterate through all combinations of parameters
# for K in K_values:
#     for L in L_values:
#         # for P in P_values:
#             # for H in H_values:
#         # Construct the command
#         command = [
#             "srun",
#             "-c", "2",
#             "--gres=gpu:1",
#             "--pty",
#             "python",
#             "-m", "hw2.experiments",
#             "run-exp",
#             "-n", "exp1_1",
#             "-K", str(K),
#             "-L", str(L),
#             "-H", "100",
#             "--early-stopping", "8",
#             "-M", "cnn",
#             "-s", "42",
#             "-P", str(min(5, int(L/2)))
#         ]
            
#         # Execute the command
#         print("Running command: {}".format(' '.join(command)))
#         subprocess.call(command)




#exp1_2
# Define the range of values for the parameters
L_values = [8]
K_values = [128]
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
            "-n", "exp1_2",
            "-K", str(K),
            "-L", str(L),
            "-H", "100",
            "--early-stopping", "8",
            "-M", "cnn",
            "-s", "42",
            "-P", str(min(5, int(L/2))),
            "--lr", str(1e-4)
        ]
        
        # Execute the command
        print("Running command: {}".format(' '.join(command)))
        subprocess.call(command)




# #exp1_3
# # Define the range of values for the parameters
# L_values = [2,3,4]
# K_values = [[32, 64]]
# # P_values = [2, 3]
# # H_values = [100, 200]

# # Iterate through all combinations of parameters
# for K in K_values:
#     for L in L_values:
#         # for P in P_values:
#             # for H in H_values:
#                 # Construct the command
#         command = [
#             "srun",
#             "-c", "2",
#             "--gres=gpu:1",
#             "--pty",
#             "python",
#             "-m", "hw2.experiments",
#             "run-exp",
#             "-n", "exp1_3",
#             "-K", "64", "128",
#             "-L", str(L),
#             "-H", "100",
#             "--early-stopping", "8",
#             "-M", "cnn",
#             "-s", "42",
#             "-P", str(min(5, int(L/2)))
#         ]
        
#         # Execute the command
#         print("Running command: {}".format(' '.join(command)))
#         subprocess.call(command)



# #exp1_4_1
# # Define the range of values for the parameters
# L_values = [8, 16, 32]
# K_values = [32]
# # P_values = [2, 3]
# # H_values = [100, 200]

# # Iterate through all combinations of parameters
# for K in K_values:
#     for L in L_values:
#         # for P in P_values:
#             # for H in H_values:
#                 # Construct the command
#         command = [
#             "srun",
#             "-c", "2",
#             "--gres=gpu:1",
#             "--pty",
#             "python",
#             "-m", "hw2.experiments",
#             "run-exp",
#             "-n", "exp1_4",
#             "-K", str(K),
#             "-L", str(L),
#             "-H", "100",
#             "--early-stopping", "8",
#             "-M", "resnet",
#             "-s", "42",
#             "-P", str(min(5, int(L/2)))  
#         ]
        
#         # Execute the command
#         print("Running command: {}".format(' '.join(command)))
#         subprocess.call(command)



# #exp1_4_2
# # Define the range of values for the parameters
# L_values = [2,4,8]
# K_values = [[64, 128, 256]]
# # P_values = [2, 3]
# # H_values = [100, 200]

# # Iterate through all combinations of parameters
# for K in K_values:
#     for L in L_values:
#         # for P in P_values:
#             # for H in H_values:
#                 # Construct the command
#         command = [
#             "srun",
#             "-c", "2",
#             "--gres=gpu:1",
#             "--pty",
#             "python",
#             "-m", "hw2.experiments",
#             "run-exp",
#             "-n", "exp1_4",
#             "-K", "64", "128", "256",
#             "-L", str(L),
#             "-H", "100",
#             "--early-stopping", "8",
#             "-M", "resnet",
#             "-s", "42",
#             "-P", str(min(5, int(L/2)))
#         ]
        
#         # Execute the command
#         print("Running command: {}".format(' '.join(command)))
#         subprocess.call(command)
