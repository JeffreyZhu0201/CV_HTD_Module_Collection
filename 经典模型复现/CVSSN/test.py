import utils.data_load_operate as data_load_operate
import os

results_save_path = \
    os.path.join(os.getcwd(), 'output/results', model_list[model_flag] + str("_") +
                 data_set_name + str("_") + str(time_current) + str("_seed") + str(seed) + str("_ratio") + str(
        ratio) + str("_patch_size") + str(patch_size))
cls_map_save_path = \
    os.path.join(os.path.join(os.getcwd(), 'output/cls_maps'), model_list[model_flag] + str("_") +
                 data_set_name + str("_") + str(time_current) + str("_seed") + str(seed)) + str("_ratio") + str(ratio)

data, gt = data_load_operate.load_data(data_set_name, data_set_path)

# get dataset information
height, width, channels = data.shape

print("data:",data)
print("data shape",data.shape)


# standardlize the dataset
data = data_load_operate.standardization(data)
print(gt)
# reshape as 
gt_reshape = gt.reshape(-1)

print(data)
print(gt)