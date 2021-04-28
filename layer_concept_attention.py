''' Layers
    This file contains various layers for the BigGAN models.
'''
import functools

import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Parameter as P
from torch.autograd import Variable
import faiss

from sync_batchnorm import SynchronizedBatchNorm2d as SyncBN2d

from torch.distributions import Categorical
from layers import SNConv2d


class ConceptAttentionProto(nn.Module):
    """concept attention"""
    def __init__(self, pool_size_per_cluster, num_k, feature_dim, warmup_total_iter=1000, device='cuda'):
        super(ConceptAttentionProto, self).__init__()
        self.device = device
        self.pool_size_per_cluster = pool_size_per_cluster
        self.num_k = num_k
        self.feature_dim = feature_dim
        self.total_pool_size = self.num_k * self.pool_size_per_cluster
        self.register_buffer('concept_pool', torch.rand(self.feature_dim, self.total_pool_size))
        self.register_buffer('concept_proto', torch.rand(self.feature_dim, self.num_k))
        # concept pool is arranged as memory cell, i.e. linearly arranged as a 2D tensor, use get_cluster_ptr to get starting pointer for each cluster
        

        # states that indicating the warmup
        self.warmup_state = 1 # 1 means during warm up, will switch to 0 after warmup
        self.warmup_iter_counter = 0
        self.warmup_total_iter = warmup_total_iter
        self.pool_structured = 0 # 0 means pool is un clustered, 1 mean pool is structured as clusters arrays

        # register attention module
        self.attention_module = ConceptAttention(self.feature_dim)

        

    def get_cluster_ptr(self, cluster_num):
        """get starting pointer for cluster_num"""
        assert cluster_num < self.num_k, f"cluster_num {cluster_num} out of bound (totally has {self.num_k} clusters)"
        return self.pool_size_per_cluster * cluster_num
    
    def _update_pool(self, index, content):
        """update concept pool according to the content"""
        assert len(index.shape) == 1
        assert content.shape[1] == index.shape[0]
        assert content.shape[0] == self.feature_dim
        
        print("Updating concept pool...")
        self.concept_pool[:, index] = content
    
    def _update_prototypes(self, index, content):
        assert len(index.shape) == 1
        assert content.shape[1] == index.shape[0]
        assert content.shape[0] == self.feature_dim
        print("Updating prototypes...")
        self.concept_proto[:, index] = content

    def pool_kmean_init(self, seed=0, gpu_num=0, temperature=1):
        """TODO: clear up
        perform kmeans for cluster concept pool initialization
        Args:
            x: data to be clustered
        """
        
        print('performing kmeans clustering')
        results = {'im2cluster':[],'centroids':[],'density':[]}
        x = self.concept_pool.clone().cpu().numpy().T
        x = np.ascontiguousarray(x)
        num_cluster = self.num_k
        # intialize faiss clustering parameters
        d = x.shape[1]
        k = int(num_cluster)
        clus = faiss.Clustering(d, k)
        clus.verbose = True
        clus.niter = 100
        clus.nredo = 10
        clus.seed = seed
        clus.max_points_per_centroid = 1000
        clus.min_points_per_centroid = 10

        res = faiss.StandardGpuResources()
        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16 = False
        cfg.device = gpu_num   
        index = faiss.GpuIndexFlatL2(res, d, cfg)  

        clus.train(x, index)   

        D, I = index.search(x, 1) # for each sample, find cluster distance and assignments
        im2cluster = [int(n[0]) for n in I]
        
        # get cluster centroids
        centroids = faiss.vector_to_array(clus.centroids).reshape(k,d)
        
        # sample-to-centroid distances for each cluster 
        Dcluster = [[] for c in range(k)]          
        for im,i in enumerate(im2cluster):
            Dcluster[i].append(D[im][0])
        
        # concentration estimation (phi)        
        density = np.zeros(k)
        for i,dist in enumerate(Dcluster):
            if len(dist)>1:
                d = (np.asarray(dist)**0.5).mean()/np.log(len(dist)+10)            
                density[i] = d     
                
        #if cluster only has one point, use the max to estimate its concentration        
        dmax = density.max()
        for i,dist in enumerate(Dcluster):
            if len(dist)<=1:
                density[i] = dmax 

        density = density.clip(np.percentile(density,10),np.percentile(density,90)) #clamp extreme values for stability
        print(density.mean())
        density = temperature*density/density.mean()  #scale the mean to temperature 
        
        # convert to cuda Tensors for broadcast
        centroids = torch.Tensor(centroids).cuda()
        centroids = nn.functional.normalize(centroids, p=2, dim=1)    

        im2cluster = torch.LongTensor(im2cluster).cuda()               
        density = torch.Tensor(density).cuda()
        
        results['centroids'].append(centroids)
        results['density'].append(density)
        results['im2cluster'].append(im2cluster)    
        
        # rearrange
        self.structure_memory_bank(results) 
        print("Finish kmean init...")


    def get_cluster_num_index(self, idx):
        assert idx < self.total_pool_size
        return idx // self.pool_size_per_cluster


    def structure_memory_bank(self, cluster_results):
        """make memory bank structured """
        centeriod = cluster_results['centroids'][0] # [num_k, feature_dim]
        cluster_assignment = cluster_results['im2cluster'][0] # [total_pool_size,]
        
        mem_index = torch.zeros(self.total_pool_size).long() # array of memory index that contains instructions of how to rearange the memory into structured clusters
        memory_states = torch.zeros(self.num_k,).long() # 0 indicate the cluster has not finished structured
        memory_cluster_insert_ptr = torch.zeros(self.num_k,).long() # ptr to each cluster block

        # loop through every cluster assignment to populate the concept pool for each cluster seperately
        for idx, i in enumerate(cluster_assignment):
            cluster_num = self.get_cluster_num_index(i)
            if memory_states[cluster_num] == 0:
                
                # manipulating the index for populating memory
                mem_index[cluster_num * self.pool_size_per_cluster + memory_cluster_insert_ptr[cluster_num]] = idx  

                memory_cluster_insert_ptr[cluster_num] += 1
                if memory_cluster_insert_ptr[cluster_num] == self.pool_size_per_cluster:
                    memory_states[cluster_num] = 1
            else:
                # check if the ptr for this class is set to the last point
                assert memory_cluster_insert_ptr[cluster_num] == self.pool_size_per_cluster
        
        # what if some cluster didn't get populated enough? -- replicate
        not_fill_cluster = torch.where(memory_states == 0)[0]
        for unfill_cluster in not_fill_cluster:
            cluster_ptr = memory_cluster_insert_ptr[unfill_cluster]
            existed_index = mem_index[unfill_cluster * self.pool_size_per_cluster : cluster_ptr]
            replicate_times = self.pool_size_per_cluster // cluster_ptr + 1 # with more replicate and cutoff
            replicated_index = torch.cat([existed_index for _ in range(replicate_times)])
            # permutate the replicate and select pool_size_per_cluster num of index
            replicated_index = replicated_index[torch.randperm(replicated_index.shape[0])][:self.pool_size_per_cluster] # [pool_size_per_cluster, ]
            # put it back
            assert replicated_index.shape[0] == self.pool_size_per_cluster, f"replicated_index ({replicated_index}) should has the same len as pool_size_per_cluster ({self.pool_size_per_cluster})"
            mem_index[unfill_cluster * self.pool_size_per_cluster: (unfill_cluster+1) * self.pool_size_per_cluster] = replicated_index
            # update ptr
            memory_cluster_insert_ptr[unfill_cluster] = self.pool_size_per_cluster
            # update state
            memory_states[unfill_cluster] = 1
        
        assert (memory_states == 0).sum() == 0, f"memory_states has zeros: {memory_states}"
        assert (memory_cluster_insert_ptr != self.pool_size_per_cluster).sum() == 0, f"memory_cluster_insert_ptr didn't match with pool_size_per_cluster: {memory_cluster_insert_ptr}"


        # update the real pool
        self._update_pool(torch.arange(mem_index.shape[0]), self.concept_pool.clone()[:, mem_index])
        # initialize the prototype
        self._update_prototypes(torch.arange(self.num_cluster), self.centeriod.T)
        print(f"Concept pool updated by kmeans clusters...")



    def _check_warmup_state(self):
        """check if need to switch warup_state to 0; when turn off warmup state, trigger k-means init for clustering"""
        if self.warmup_state:
            if self.warmup_iter_counter > self.warmup_total_iter:
                self.warmup_state = 0 # 0 means not in a warmup state
                # trigger kmean concept pool init
                self.pool_kmean_init()
                
        else:
            raise Exception("Calling _check_warmup_state when self.warmup_state is 0 (0 means not in warmup state)")


    def warmup_sampling(self, x):
        """
        linearly sample input x to make it 
        x: [n, c, h, w]"""
        n, c, h, w = shape
        assert self.warmup_state, "calling warmup sampling when warmup state is 0"
        
        # evenly distributed across space
        sample_per_instance = max(int(self.total_pool_size / n), 1)
        
        # sample index
        index = torch.randint(h * w, size=(n, 1, sample_per_instance)).repeat(1, c, 1).to(self.device) # n, c, sample_per_instance
        sampled_columns = torch.gather(x.reshape(n, c, h * w), 2, index) # n, c, sample_per_instance 
        sampled_columns = torch.transpose(sampled_columns, 1, 0).reshape(c, -1).contiguous() # c, n * sample_per_instance
        
        # calculate percentage to populate into pool, as the later the better, use linear intepolation from 1% to 50% according to self.warmup_iter_couunter
        percentage = (self.warmup_iter_counter + 1) / self.warmup_total_iter * 0.5 # max percent is 50%
        sample_column_num = max(1, int(percentage * sampled_columns.shape[1]))
        sampled_columns_idx = torch.randint(sampled_columns.shape[1], size=(sample_column_num,))
        sampled_columns = sampled_columns[:, sampled_columns_idx]  # [c, sample_column_num]

        # random select pool idx to update
        update_idx = torch.randperm(self.concept_pool.shape[1])[:sample_column_num]
        self._update_pool(update_idx, sampled_columns)

        # update number
        self.warmup_iter_counter += 1


    def forward(self, x, device="cuda"):
        n, c, h, w = x.shape 

        # warmup
        if self.warmup_state:
            # if still in warmup, skip attention
            self.warmup_sampling(x)
            self._check_warmup_state()
            return x

        else:
            # attend to concepts
            # selecting 
            

            

        
         



# the original attention was self-attention
# here is an attention module to be used by memory bank
class ConceptAttention(nn.Module):
    def __init__(self, ch, which_conv=SNConv2d):
        super(ConceptAttention, self).__init__()
        self.myid = "atten"
        # Channel multiplier
        self.ch = ch
        self.which_conv = which_conv
        self.theta = self.which_conv(
            self.ch, self.ch // 2, kernel_size=1, padding=0, bias=False)
        self.phi = self.which_conv(
            self.ch, self.ch // 2, kernel_size=1, padding=0, bias=False)
        self.g = self.which_conv(
            self.ch, self.ch // 2, kernel_size=1, padding=0, bias=False)
        self.o = self.which_conv(
            self.ch // 2, self.ch, kernel_size=1, padding=0, bias=False)
        # Learnable gain parameter
        self.gamma = P(torch.tensor(0.), requires_grad=True)

    def forward(self, x, y):
        # x from encoder (N x 128 x 128), 64 (64 is the feature map size of that layer
        # y from memory bank 512, 64
        # Apply convs
        theta = self.theta(x)
        phi = self.phi(y)
        g = self.g(y)
        # phi = F.max_pool2d(self.phi(x), [2, 2])
        # g = F.max_pool2d(self.g(x), [2, 2])
        # Perform reshapes
        theta = theta.view(-1, self.ch // 2, x.shape[2] * x.shape[3])
        phi = phi.view(-1, self.ch // 2, y.shape[0])
        g = g.view(-1, self.ch // 2, y.shape[0])
        # Matmul and softmax to get attention maps
        beta = F.softmax(torch.bmm(theta.transpose(1, 2), phi), -1)
        # Attention map times g path
        o = self.o(torch.bmm(g, beta.transpose(1, 2)).view(-1,
                                                           self.ch // 2, x.shape[2], x.shape[3]))
        return self.gamma * o + x


class MemoryClusterAttention(nn.Module):
    def __init__(self, dim=64, n_embed=512):
        super(MemoryClusterAttention, self).__init__()

        self.dim = dim   # set to 64 currently
        self.n_embed = n_embed   # vq vae use 512, maybe we can use the same?

        embed = torch.randn(n_embed, dim)
        self.register_parameter("embed", embed)
        self.attention_module = ConceptAttention(dim)

        # currently dont need them as we don't have multiple clusters
        # self.register_buffer("cluster_size", torch.zeros(n_embed))
        # self.register_buffer("embed_avg", embed.clone())

    def attend_to_memory_bank(self, x):
        return self.attention_module.forward(x, self.embed)

    def forward(self, input: torch.Tensor):
        # input_dim [batch_size, emb_dim, h, w]
        return self.attend_to_memory_bank(input)



if __name__ == '__main__':
    concept_atten_proto = ConceptAttentionProto(pool_size_per_cluster=100, num_k=20, feature_dim=128)
    results_kmean = concept_atten_proto.pool_kmean_init()