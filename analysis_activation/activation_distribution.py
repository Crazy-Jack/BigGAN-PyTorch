import numpy as np
import matplotlib.pyplot as plt  
import copy

plt.rcParams['agg.path.chunksize'] = 10000

ACTIVATION_PATH = '../samples/test_test_VAE18_GAN_selfatt_get_activation/inter_activation_iter_{}.npy'
ITER_NUM = 20
criticals = [0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.7]


# # aggregate all 10,000 images
activation = np.load(ACTIVATION_PATH.format(1), allow_pickle=True).item()
BATCH_SIZE = activation[1].shape[0]

critic_indx_layer_0 = np.load("critic_idnx_layer_0.npy")
num_critics = critic_indx_layer_0.shape[1]

critical_activation = {i: np.zeros((BATCH_SIZE * ITER_NUM, num_critics)) for i in activation}
print("total instance ", BATCH_SIZE * ITER_NUM)
critic_indx = {}
for i in activation:
    critic_indx[i] = np.load("critic_idnx_layer_{}.npy".format(i))


# print(activation[1].shape)
    
for iter_ in range(1, ITER_NUM + 1):
    activation_i = np.load(ACTIVATION_PATH.format(iter_), allow_pickle=True).item()
    for i in activation:
        C_idx, H_idx, W_idx = critic_indx[i]
        critic_activation_i = np.abs(activation_i[i])[:, C_idx, H_idx, W_idx] # [BZ, num_critics]
        critical_activation[i][(iter_ - 1) * BATCH_SIZE : iter_ * BATCH_SIZE] = critic_activation_i # (BZ, num_critics)
        print("adding for {}: {}".format((iter_ - 1) * BATCH_SIZE, iter_ * BATCH_SIZE))
    print("Done Loading with iteration {} ---".format(iter_))

np.save("critical_activation.npy", critical_activation) # {layer_num: [BZ, num_critics]}

# critical_activation = np.load("critical_activation.npy", allow_pickle=True)

for i in critical_activation: # loop over layers
    for j in range(num_critics):
        plt.clf()
        plt.plot(critical_activation[i][:, j])
        plt.xlabel("picture number")
        plt.ylabel("activation for this neuron")
        plt.title("activation for one neuron in layer {} ({} % th)".format(i, criticals[j] * 100))
        plt.savefig("activation_per_neurons_layer_{}_num_critic_{}.png".format(i, j))

        plt.clf()
        plt.plot(sorted(critical_activation[i][:, j])[::-1])
        plt.xlabel("picture number")
        plt.ylabel("activation for this neuron")
        plt.title("activation for one neuron in layer {} ({} % th)".format(i, criticals[j]))
        plt.savefig("sorted_activation_per_neurons_layer_{}_num_critic_{}.png".format(i, j))

        plt.close()


