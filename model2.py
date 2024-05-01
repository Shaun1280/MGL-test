import torch
import torch.nn as nn
import torch.nn.functional as functional
import numpy as np
from collections import defaultdict
from copy import deepcopy
import torch_sparse
import random

class EmbeddingGenerator(nn.Module):
    def __init__(self, user_num, item_num, item_feature_list, item_feature_matrix, dense_f_list_transforms, opt, device):
        super().__init__()

        self.user_num = user_num
        self.item_num = item_num

        # Copy the item feature list to avoid modifying the original
        self.item_feature_list = deepcopy(item_feature_list)
        self.item_feature_matrix = item_feature_matrix.to(device)

        # Move each dense feature to the specified device
        self.item_dense_features = [dense_f.to(device) for dense_f in dense_f_list_transforms.values()]

        # Remove the 'encoded' feature from the item feature list
        self.item_feature_list.remove({'feature_name':'encoded', 'feature_dim':self.item_num})

        # store the embedding dimensions for each feature
        item_embedding_dims = defaultdict(int)
        for f in self.item_feature_list:
            item_embedding_dims[f['feature_name']] = opt.embedding_size

        # Calculate the total embedding dimension
        self.item_total_emb_dim = sum(item_embedding_dims.values()) + opt.dense_embedding_dim * len(self.item_dense_features)

        # Create an embedding layer for each item feature
        self.item_Embeddings = nn.ModuleList([nn.Embedding(feature['feature_dim'], item_embedding_dims[feature['feature_name']]) for feature in self.item_feature_list])

        # Create a linear layer for each dense feature
        self.item_dense_Embeddings = nn.ModuleList([nn.Linear(dense_f.shape[1], opt.dense_embedding_dim, bias=False) for dense_f in self.item_dense_features])

        # Define the encoder and decoder networks
        self.encoder = nn.Sequential(nn.Linear(self.item_total_emb_dim, 256, bias=True), nn.ReLU(), nn.Linear(256, 64, bias=True), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(64, 256, bias=True), nn.ReLU(), nn.Linear(256, opt.id_embedding_size, bias=True))

    def encode(self, item_id):
        # Embed the item features and pass them through the encoder
        return self.encoder(self.embed_feature(item_id))

    def decode(self, batch_item_feature_encoded):
        # Pass the encoded features through the decoder
        return self.decoder(batch_item_feature_encoded)

    def embed_feature(self, item_id):
        # Embed each item feature and concatenate them
        batch_item_feature_embedding = []
        batch_item_feature  = self.item_feature_matrix[item_id]
        for i, _ in enumerate(self.item_feature_list):
            embedding_layer = self.item_Embeddings[i]
            batch_item_feature_i = batch_item_feature[:, i]
            batch_item_feature_i_embedding = embedding_layer(batch_item_feature_i)
            batch_item_feature_embedding.append(batch_item_feature_i_embedding)

        batch_item_feature_embedding = torch.cat(batch_item_feature_embedding, -1)

        # Embed each dense feature and concatenate them with the item features
        dense_embeddings = []
        for i, dense_f in enumerate(self.item_dense_features):
            batch_dense_f = dense_f[item_id]
            embedding_layer = self.item_dense_Embeddings[i]
            dense_embedding = embedding_layer(batch_dense_f.float()) / torch.sum(batch_dense_f.float(), dim=1, keepdim=True)
            dense_embeddings.append(dense_embedding)

        batch_item_feature_embedding = torch.cat([batch_item_feature_embedding] + dense_embeddings, dim=1)

        return batch_item_feature_embedding
    
class Model(nn.Module):
    def __init__(self, Data, opt, device):
        super().__init__()

        self.name = "MGL Reimplementation"

        self.device = device

        self.L = opt.L
        self.convergence = opt.convergence
        self.link_topk = opt.link_topk
        self.interact_train = Data.interact_train

        self.user_num = Data.user_num
        self.user_degrees = Data.user_degrees 
        self.user_id_Embeddings = nn.Embedding(self.user_num, opt.id_embedding_size)

        self.item_num = Data.item_num
        self.item_degrees = Data.item_degrees
        self.item_id_Embeddings = nn.Embedding(self.item_num, opt.id_embedding_size)

        # sort by id
        self.sorted_item_degrees = sorted(self.item_degrees.items(), key=lambda x: x[0])
        _, self.item_degree_list = zip(*self.sorted_item_degrees)

        # sort by degree
        self.sorted_item_degrees = sorted(self.item_degrees.items(), key=lambda x: x[1])
        sorted_item_list, _= zip(*self.sorted_item_degrees)
        self.top_rate = opt.top_rate
        self.top_length = int(self.item_num * self.top_rate)
        self.top_item = torch.tensor(sorted_item_list[-self.top_length:]).to(self.device)
        self.sorted_item = torch.tensor(sorted_item_list).to(self.device)

        self.generator = EmbeddingGenerator(self.user_num, self.item_num, Data.item_feature_list, Data.item_feature_matrix, Data.dense_f_list_transforms, opt, device)
        self._create_adjacency_matrix()


    def _create_adjacency_matrix(self):
        index = [self.interact_train['userid'].tolist(), self.interact_train['itemid'].tolist()]
        value = [1.0] * len(self.interact_train)

        # R in 3
        self.interact_matrix = torch.sparse_coo_tensor(index, value, (self.user_num, self.item_num)).to(self.device)

        index = [self.interact_train['userid'].tolist(), (self.interact_train['itemid'] + self.user_num).tolist()]
        matrix_size = self.user_num + self.item_num
        adjacency_matrix = torch.sparse_coo_tensor(index, value, (matrix_size, matrix_size))
        
        # A_G in 3
        adjacency_matrix = (adjacency_matrix + adjacency_matrix.t()).coalesce()
        
        row_indices, col_indices = adjacency_matrix.indices()[0], adjacency_matrix.indices()[1]
        adjacency_matrix_value = adjacency_matrix.values()

        norm_deg = torch.pow(torch.sparse.sum(adjacency_matrix, dim=1).to_dense() + 1, -0.5)
        norm_deg2 = torch.pow(torch.sparse.sum(adjacency_matrix, dim=0).to_dense() + 1, -0.5)

        adjacency_matrix_norm_value = norm_deg[row_indices] * adjacency_matrix_value * norm_deg2[col_indices]

        # nomalized A_G
        self.adjacency_matrix_normed = torch.sparse_coo_tensor(torch.stack([row_indices, col_indices], dim=0), \
                                                                      adjacency_matrix_norm_value, (matrix_size, matrix_size)).to(self.device)


    def _gcn(self, row_index, colomn_index, s_hat):
        indice = torch.cat([row_index, colomn_index], dim=0).to(self.device)
        
        # equation (14)
        cur_embedding = torch.cat([self.user_id_Embeddings.weight, self.item_id_Embeddings.weight], dim=0)
        all_embeddings = [cur_embedding]

        matrix_size = self.user_num + self.item_num

        for _ in range(self.L):
            # equation (3)
            # contribution of A_G without S_hat
            original_embedding = torch.mm(self.adjacency_matrix_normed.to_dense(), cur_embedding)

            # contribution of S_hat
            enhanced_embedding = torch_sparse.spmm(indice, s_hat, matrix_size, matrix_size, cur_embedding)

            # sum up
            cur_embedding = original_embedding + enhanced_embedding

            all_embeddings.append(cur_embedding)

        all_embeddings = torch.stack(all_embeddings, dim=0)
        all_embeddings = torch.mean(all_embeddings, dim=0)

        return all_embeddings


    # see equation (12) and (13)
    def _s_hat_sparse(self, top_item_embedding, sorted_item_embedding):
        s = torch.mm(sorted_item_embedding, top_item_embedding.t())

        # [item_num, link_topk]
        s_masked, indices = s.topk(self.link_topk, dim=-1)
        s_masked = s_masked.sigmoid()

        sorted_item_index = self.sorted_item.unsqueeze(1).expand_as(s).gather(1, indices).reshape(-1)
        # for every item in sparse representation
        row_index = (sorted_item_index + self.user_num).unsqueeze(0)

        top_item_index = self.top_item.unsqueeze(0).expand_as(s).gather(1, indices).reshape(-1)
        colomn_index = (top_item_index + self.user_num).unsqueeze(0)

        # regularization
        degree_norm = torch.pow(torch.sum(s_masked, dim=1) + 1, -0.5).unsqueeze(1).expand_as(s_masked).reshape(-1)
        degree_norm2 = torch.pow(torch.sum(s_masked, dim=0) + 1, -0.5).unsqueeze(0).expand_as(s_masked).reshape(-1)
        enhanced_value = s_masked.reshape(-1)
        s_hat = enhanced_value * degree_norm * degree_norm2
        
        return row_index, colomn_index, s_hat


    def _forward_theta(self, theta):
        encoder_0_weight = theta[0]
        encoder_0_bias = theta[1]
        encoder_2_weight = theta[2]
        encoder_2_bias = theta[3]

        top_item_feature = self.generator.embed_feature(self.top_item)
        sorted_item_feature = self.generator.embed_feature(self.sorted_item)

        top_item_hidden = torch.mm(top_item_feature, encoder_0_weight.t()) + encoder_0_bias
        top_item_embedding = torch.mm(top_item_hidden, encoder_2_weight.t()) + encoder_2_bias

        sorted_item_hidden = torch.mm(sorted_item_feature, encoder_0_weight.t()) + encoder_0_bias
        sorted_item_embedding = torch.mm(sorted_item_hidden, encoder_2_weight.t()) + encoder_2_bias

        return top_item_embedding, sorted_item_embedding


    # see 4.1 L_GL
    def gl_loss(self, item1, item2):
        mse_loss = nn.MSELoss()
        item1_aux_embedding = self.generator.encode(item1)
        item2_aux_embedding = self.generator.encode(item2) # batch size * d

        item_list = list(range(self.item_num))
        random.shuffle(item_list)
        batch_size = item2_aux_embedding.shape[0]
        item_neg = torch.tensor(item_list[:batch_size]).to(self.device)
        item_neg_aux_embedding = self.generator.encode(item_neg)

        score = torch.mm(item1_aux_embedding, item2_aux_embedding.t()).sigmoid()
        score_neg = torch.mm(item1_aux_embedding, item_neg_aux_embedding.t()).sigmoid()

        return (mse_loss(score, torch.ones_like(score)) + mse_loss(score_neg, torch.zeros_like(score_neg))) / 2


    # see 4.2 L_PCL
    def pcl_loss(self, observed_item):
        observed_item_aux_embedding = self.generator.encode(observed_item)
        observed_item_org_embedding = self.generator.decode(observed_item_aux_embedding)

        def pop(x, k):
            return 1 - (k / (k + np.exp(x / k)))

        # equation(11)
        item_degree = np.array(self.item_degree_list)[observed_item.cpu().numpy()]
        item_pop = pop(item_degree, self.convergence)
        
        # this determines whether the pcl loss will be omitted
        keep = torch.distributions.binomial.Binomial(1, torch.from_numpy(item_pop)).sample().to(self.device)

        l_pcl = functional.mse_loss(observed_item_org_embedding, self.item_id_Embeddings(observed_item), reduction='none').mean(dim=-1) 
        
        term_count = keep.sum()

        if term_count.item() == 0:
            return 0 * torch.mul(keep, l_pcl).sum()
        
        l_pcl = torch.mul(keep, l_pcl).sum() / term_count # taking the average over a batch
        return l_pcl

    def rec_loss(self, user_id, observed_item, unobserved_item, theta):
        top_item_embedding, sorted_item_embedding = self._forward_theta(theta)

        # sparse representation of S hat 
        row_index, colomn_index, joint_enhanced_value = self._s_hat_sparse(top_item_embedding, sorted_item_embedding)
        
        all_embeddings = self._gcn(row_index, colomn_index, joint_enhanced_value)

        user_embeddings, item_embeddings = torch.split(all_embeddings, [self.user_num, self.item_num])

        # equation (4)
        user_embedding = user_embeddings[user_id]
        observed_score = torch.mul(user_embedding, item_embeddings[observed_item]).sum(dim=-1)
        unobserved_score = torch.mul(user_embedding, item_embeddings[unobserved_item]).sum(dim=-1)
        
        # rec loss
        return -(observed_score - unobserved_score).sigmoid().log().mean() 


    def predict(self, user_id):
        top_item_embedding = self.generator.encode(self.top_item)
        sorted_item_embedding = self.generator.encode(self.sorted_item)

        # sparse representation of S hat 
        row_index, colomn_index, s_hat = self._s_hat_sparse(top_item_embedding, sorted_item_embedding)

        all_embeddings = self._gcn(row_index, colomn_index, s_hat)

        user_embeddings, item_embeddings = torch.split(all_embeddings, [self.user_num,self.item_num])

        return torch.mm(user_embeddings[user_id], item_embeddings.t())