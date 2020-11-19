import numpy as np
import matplotlib.pyplot as plt  

ACTIVATION_PATH = '../samples/test_test_VAE18_GAN_selfatt_get_activation/inter_activation_iter_{}.npy'
ITER_NUM = 20


# aggregate all 10,000 images
activation = np.load(ACTIVATION_PATH.format(1), allow_pickle=True).item()

layer_half_act = {i: None for i in activation}
# mean
for i in activation:
    t = 0.5 * np.max(activation[i].reshape(activation[i].shape[0], -1))
    layer_half_act[i] = 
print("Done Loading with iteration {} ---".format(1))






for iter_ in range(2, ITER_NUM):
    activation_i = np.load(ACTIVATION_PATH.format(iter_), allow_pickle=True).item()
    for i in activation_i:
        activation[i] += activation_i[i].mean(0) # (C, H, W)
    print("Done Loading with iteration {} ---".format(iter_))

for i in activation:
    activation[i] /= ITER_NUM
np.save("mean_response.npy", activation)




# activation = np.load("mean_response.npy", allow_pickle=True).item()
for i in activation:
    print("mean activation for layer {}: {}".format(i, activation[i].shape))
    plot_sum_act(activation[i], "avg_activation_layer_{}_({})".format(i, activation[i].shape))




# for i in activation:
#     print("On layer {}: {}".format(i, activation[i].shape))
#     activation_shape = activation[i].shape
#     # find top1/top5/10/25/50 percent threshold
#     t = 0.0

#     # # mask out non essential activation
#     # mask_index = np.where(np.abs(activation[i]) > t)
#     # mask = np.zeros_like(activation[i])
#     # mask[mask_index] = 1.0
#     # act_t = mask * activation[i]
#     # print("get masked activation with threshold {}".format(t))
#     act_t = activation[i]

#     # sum of activation
#     sum_act_t = act_t.sum(0) # (C, H, W)
#     plot_sum_act(sum_act_t, "sum_activation_layer_{} ({})".format(i, activation_shape))





