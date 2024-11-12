from status_control import *

reg_type = ["ExtraTreesRegressor", "KnnRegression", "VotingRegressor", "Linear", "RandomForestRegressor", "GradientBoostingRegressor", "SupportVectorRegressor", "GradientDescentRegression"]
batch_size_option = [256, 128, 64]
epoch_option = [50, 100, 150, 200, 250, 300]

# build_status("GradientDescentRegression", batch_size_option, epoch_option)
# manuel_modify_status()
status_print()
# a=check_status("VotingRegressor", 256, 50)
# print(type(a))