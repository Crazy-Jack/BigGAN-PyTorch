import torch 
import torch.nn as nn 
import torchvision 
import os

class LinearCombineVC(nn.Module):
    """linearly combine visual concepts together to reconstruct the feature map directly"""
    def __init__(self, topk, visual_concept_pool_size, visual_concept_dim, mode, lambda_l1_reg_dot=1, test=False):
        super(LinearCombineVC, self).__init__()
        self.myid = "linear_comb_vc"
        self.visual_concept_dim = visual_concept_dim # should be the same as channels
        self.visual_concept_pool_size = visual_concept_pool_size
        self.mode = mode
        self.topk = topk
        self.lambda_l1_reg_dot = lambda_l1_reg_dot
        self.visual_concept_pool = nn.Parameter(torch.rand(self.visual_concept_pool_size, self.visual_concept_dim), requires_grad=True)
        self.test = test
        print(f"^^^^^^^^^^^test {test}")

    def forward(self, x, device="cuda"):
        n, c, h, w = x.shape 
        # compute similarity for every position
        dot_product = torch.matmul(self.visual_concept_pool[None, :, :], x.reshape(n, c, h * w)) # [1, pool_size, c] x [n, c, h * w] -> [n, pool_size, h * w]
        norm_product = torch.matmul((self.visual_concept_pool**2).sum(1).sqrt()[None, :, None], (x.reshape(n, c, h * w)**2).sum(1, keepdim=True).sqrt()) # [n, pool_size, h * w]
        cosine_similarity = dot_product / norm_product # [n, pool_size, h*w]
        
        if float(self.mode) == 1.0:
            # reconstruct
            reconstruct = torch.matmul(torch.transpose(self.visual_concept_pool[None, :, :], 1, 2), cosine_similarity) # [n, c, h * w]
            # print out how many zero cosine_similarity
            # print(f"(cosine_similarity == 0.)*1. {(cosine_similarity == 0.)*1.}")
            print(f"num of zeros {((cosine_similarity == 0.)*1.).sum()}")
            regularization = cosine_similarity.abs().mean() * self.lambda_l1_reg_dot
            
            if self.test:
                # plot the cosine similarity map
                # save_path = "/lab_data/leelab/tianqinl/BigGAN-PyTorch/scripts/celeba/evals/eval_linear_comb_1.0"
                # cosine_similarity_visualization = cosine_similarity.reshape(n, -1, 1, h, w)
                # cosine_similarity_visualization = cosine_similarity_visualization.cpu().repeat(1, 1, 3, 1, 1)
                # for i in range(n):
                #     imagename = os.path.join(save_path, f"visualize_vc_cosine_sim_image_{i}.png")
                #     print(f"imagename {imagename}")
                #     cosine_similarity_visualization_i = cosine_similarity_visualization[i]
                #     rank_sum = cosine_similarity_visualization_i.sum((1,2,3)) # pool_size
                #     _, index = torch.topk(rank_sum, 100, dim=0)
                #     cosine_similarity_visualization_i = cosine_similarity_visualization_i[index]
                #     torchvision.utils.save_image(cosine_similarity_visualization_i.float().cpu(), imagename,
                #                 nrow=int(cosine_similarity_visualization_i.shape[0] ** 0.5), normalize=False)

                save_path = "/lab_data/leelab/tianqinl/BigGAN-PyTorch/scripts/celeba/evals/eval_linear_comb_1.0_group_by_vc"
                os.makedirs(save_path, exist_ok=True)
                # transpose the matrix for every vc and select the top involved vc
                vc_cosine_sim = torch.transpose(cosine_similarity, 0, 1).reshape(-1, n, 1, h, w).repeat(1, 1, 3, 1, 1) # pool_size, n, 3, h, w
                # select top vc for display
                selection_criterion = vc_cosine_sim.sum((1, 2, 3, 4)) # pool size
                _, index = torch.topk(selection_criterion, 100, dim=0)
                vc_cosine_sim_select = vc_cosine_sim[index] # [select_num, n, 3, h, w]
                vc_cosine_sim_select = vc_cosine_sim_select.reshape(-1, 3, h, w) # [n * pool_size, 3, h, w]
                # normalization in non zero pixels
                min_value = vc_cosine_sim_select.reshape(-1, 3, h * w).min(2)[0].unsqueeze(-1).unsqueeze(-1)
                max_value = vc_cosine_sim_select.reshape(-1, 3, h * w).max(2)[0].unsqueeze(-1).unsqueeze(-1)
                vc_cosine_sim_select = (vc_cosine_sim_select - min_value ) / (max_value - min_value)
                # print(f"max {vc_cosine_sim_select.max()}")
                # vc_cosine_sim_select = (vc_cosine_sim_select * 255).long()
                imagename = os.path.join(save_path, f"visualize_topvc_100.png")
                torchvision.utils.save_image(vc_cosine_sim_select.float().cpu(), imagename,
                                nrow=n, normalize=False)
                print("check ")
                print(vc_cosine_sim_select[10, 0, :, :])
                



            
            return reconstruct.reshape(n, c, h, w), regularization

            
        elif float(self.mode) == 1.1:
            keep_top_num = int(self.topk * self.visual_concept_pool_size)
            _, index = torch.topk(cosine_similarity.abs(), keep_top_num, dim=1)
            mask = torch.zeros_like(cosine_similarity).scatter_(1, index, 1).to(device)
            # print out how many zero cosine_similarity
            # print(f"(cosine_similarity == 0.)*1. {(cosine_similarity == 0.)*1.}")
            
            cosine_similarity = cosine_similarity * mask
            cosine_similarity = cosine_similarity / cosine_similarity.sum(1, keepdim=True)
            reconstruct = torch.matmul(torch.transpose(self.visual_concept_pool[None, :, :], 1, 2), cosine_similarity) # [n, c, h * w]
            if self.test:
                # plot the cosine similarity map
                # save_path = "/lab_data/leelab/tianqinl/BigGAN-PyTorch/scripts/celeba/evals/eval_linear_comb_1.1"
                # cosine_similarity_visualization = cosine_similarity.reshape(n, -1, 1, h, w)
                # cosine_similarity_visualization = cosine_similarity_visualization.cpu().repeat(1, 1, 3, 1, 1)
                # for i in range(n):
                #     imagename = os.path.join(save_path, f"visualize_vc_cosine_sim_image_{i}.png")
                #     print(f"imagename {imagename}")
                #     cosine_similarity_visualization_i = cosine_similarity_visualization[i]
                #     rank_sum = cosine_similarity_visualization_i.sum((1,2,3)) # pool_size
                #     _, index = torch.topk(rank_sum, 100, dim=0)
                #     cosine_similarity_visualization_i = cosine_similarity_visualization_i[index]
                #     print(f"mean {cosine_similarity_visualization_i.mean()}")
                #     print(f"non zero mean {cosine_similarity_visualization_i.sum() / keep_top_num}")
                #     torchvision.utils.save_image(cosine_similarity_visualization_i.float().cpu(), imagename,
                #                 nrow=int(cosine_similarity_visualization_i.shape[0] ** 0.5), normalize=True)
                save_path = "/lab_data/leelab/tianqinl/BigGAN-PyTorch/scripts/celeba/evals/eval_linear_comb_1.1"
                os.makedirs(save_path, exist_ok=True)
                # transpose the matrix for every vc and select the top involved vc
                vc_cosine_sim = torch.transpose(cosine_similarity, 0, 1).reshape(-1, n, 1, h, w).repeat(1, 1, 3, 1, 1) # pool_size, n, 3, h, w
                # select top vc for display
                selection_criterion = vc_cosine_sim.sum((1, 2, 3, 4)) # pool size
                _, index = torch.topk(selection_criterion, 100, dim=0)
                vc_cosine_sim_select = vc_cosine_sim[index] # [select_num, n, 3, h, w]
                vc_cosine_sim_select = vc_cosine_sim_select.reshape(-1, 3, h, w) # [n * pool_size, 3, h, w]
                # normalization in non zero pixels
                min_value = vc_cosine_sim_select.reshape(-1, 3, h * w).min(2)[0].unsqueeze(-1).unsqueeze(-1)
                max_value = vc_cosine_sim_select.reshape(-1, 3, h * w).max(2)[0].unsqueeze(-1).unsqueeze(-1)
                vc_cosine_sim_select = (vc_cosine_sim_select - min_value ) / (max_value - min_value)
                # print(f"max {vc_cosine_sim_select.max()}")
                # vc_cosine_sim_select = (vc_cosine_sim_select * 255).long()
                imagename = os.path.join(save_path, f"visualize_topvc_100.png")
                torchvision.utils.save_image(vc_cosine_sim_select.float().cpu(), imagename,
                                nrow=n, normalize=True)
                print("check ")
                print(vc_cosine_sim_select[10, 0, :, :])
            return reconstruct.reshape(n, c, h, w), None  
        else:
            raise NotImplementedError





