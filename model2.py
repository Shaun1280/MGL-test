import torch
import torch.nn as nn
import torch.nn.functional as functional
import numpy as np
from collections import defaultdict
from copy import deepcopy
import torch_sparse
import random

class Generator(nn.Module):
    def __init__(self, user_num, item_num, item_feature_list, item_feature_matrix, dense_f_list_transforms, opt, device):
        super().__init__()

        self.user_num = user_num
        self.item_num = item_num

        self.item_feature_list = deepcopy(item_feature_list)
        self.item_feature_matrix = item_feature_matrix.to(device)

        self.item_dense_features = []
        for dense_f in dense_f_list_transforms.values():
            self.item_dense_features.append(dense_f.to(device))

        self.item_feature_list.remove({'feature_name':'encoded', 'feature_dim':self.item_num})

        item_embedding_dims = defaultdict(int)
        for f in self.item_feature_list:
            item_embedding_dims[f['feature_name']] = opt.embedding_size

        self.item_total_emb_dim = sum(list(item_embedding_dims.values())) + opt.dense_embedding_dim * len(self.item_dense_features)

        self.item_Embeddings = nn.ModuleList([nn.Embedding(feature['feature_dim'], item_embedding_dims[feature['feature_name']]) for feature in self.item_feature_list])

        self.item_dense_Embeddings = nn.ModuleList([nn.Linear(dense_f.shape[1], opt.dense_embedding_dim, bias=False) for dense_f in self.item_dense_features])

        self.encoder = nn.Sequential(nn.Linear(self.item_total_emb_dim, 256, bias=True), nn.ReLU(), nn.Linear(256, 64, bias=True), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(64, 256, bias=True), nn.ReLU(), nn.Linear(256, opt.id_embedding_size, bias=True))


    def encode(self, item_id):
        batch_item_feature_embedded = self.embed_feature(item_id)

        batch_item_feature_encoded = self.encoder(batch_item_feature_embedded)

        return batch_item_feature_encoded

    def decode(self, batch_item_feature_encoded):
        pre_item_id_embedded = self.decoder(batch_item_feature_encoded)
        return pre_item_id_embedded


    def embed_feature(self, item_id):
        batch_item_feature_embedded = []
        batch_item_feature  = self.item_feature_matrix[item_id]
        for i, f in enumerate(self.item_feature_list):
            embedding_layer = self.item_Embeddings[i]
            batch_item_feature_i = batch_item_feature[:, i]
            batch_item_feature_i_embedded = embedding_layer(batch_item_feature_i)

            batch_item_feature_embedded.append(batch_item_feature_i_embedded)

        batch_item_feature_embedded = torch.cat(batch_item_feature_embedded, -1)

        dense_embeddings = []
        for i, dense_f in enumerate(self.item_dense_features):
            batch_dense_f = dense_f[item_id]
            dense_embedded = self.item_dense_Embeddings[i](batch_dense_f.float()) / torch.sum(batch_dense_f.float(), dim = 1, keepdim= True)
            dense_embeddings.append(dense_embedded)

        batch_item_feature_embedded = torch.cat([batch_item_feature_embedded] + dense_embeddings, dim=1)

        return batch_item_feature_embedded

class Model(nn.Module):
    def __init__(self, Data, opt, device):
        super().__init__()

        self.name = "MGL Reimplementation"

        self.interact_train = Data.interact_train

        self.user_num = Data.user_num
        self.user_degrees = Data.user_degrees 
        self.user_id_Embeddings = nn.Embedding(self.user_num, opt.id_embedding_size)

        self.item_num = Data.item_num
        self.item_degrees = Data.item_degrees
        self.item_id_Embeddings = nn.Embedding(self.item_num, opt.id_embedding_size)
        self.item_feature_list = Data.item_feature_list
        self.item_feature_matrix = Data.item_feature_matrix
        
        self.dense_f_list_transforms = Data.dense_f_list_transforms
        self.L = opt.L
        self.link_topk = opt.link_topk
        self.top_rate = opt.top_rate
        self.convergence = opt.convergence

        sorted_item_degrees = sorted(self.item_degrees.items(), key=lambda x: x[0])
        _, item_degree_list = zip(*sorted_item_degrees)
        self.item_degree_numpy = np.array(item_degree_list)

        self.device = device
        self.generator = Generator(self.user_num, self.item_num, self.item_feature_list, self.item_feature_matrix, self.dense_f_list_transforms, opt, device)
        self.create_adjacency_matrix()

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

        score = torch.mm(item1_aux_embedding, item2_aux_embedding.permute(1, 0)).sigmoid()
        score_neg = torch.mm(item1_aux_embedding, item_neg_aux_embedding.permute(1, 0)).sigmoid()

        loss = (mse_loss(score, torch.ones_like(score)) + mse_loss(score_neg, torch.zeros_like(score_neg))) / 2

        return loss


    # see 4.2 L_PCL
    def pcl_loss(self, observed_item):
        observed_item_aux_embedding = self.generator.encode(observed_item)
        observed_item_org_embedding = self.generator.decode(observed_item_aux_embedding)

        def pop(x, k):
            p = 1 - (k / (k + np.exp(x / k)))
            return p

        # equation(11)
        item_degree = self.item_degree_numpy[observed_item.cpu().numpy()]
        item_pop = pop(item_degree, self.convergence)
        
        # this determines whether the pcl loss will be omitted
        m = torch.distributions.binomial.Binomial(1, torch.from_numpy(item_pop)).sample().to(self.device)

        l_pcl = functional.mse_loss(observed_item_org_embedding, self.item_id_Embeddings(observed_item), reduction='none').mean(dim=-1, keepdim=False) 
        
        if m.sum().item() == 0:
            return 0 * torch.mul(m, l_pcl).sum()
        
        l_pcl = torch.mul(m, l_pcl).sum() / m.sum() # taking the average over a batch
        return l_pcl
        

    def create_adjacency_matrix(self):
        index = [self.interact_train['userid'].tolist(), self.interact_train['itemid'].tolist()]
        value = [1.0] * len(self.interact_train)

        self.interact_matrix = torch.sparse_coo_tensor(index, value, (self.user_num, self.item_num)).to(self.device)

        index = [self.interact_train['userid'].tolist(), (self.interact_train['itemid'] + self.user_num).tolist()]
        matrix_size = self.user_num + self.item_num
        adjacency_matrix = torch.sparse_coo_tensor(index, value, (matrix_size, matrix_size))
        
        adjacency_matrix = (adjacency_matrix + adjacency_matrix.t()).coalesce()
        self.adjacency_matrix = adjacency_matrix.to(self.device)
        
        row_indices, col_indices = adjacency_matrix.indices()[0], adjacency_matrix.indices()[1]
        adjacency_matrix_value = adjacency_matrix.values()

        norm_w = torch.pow(torch.sparse.sum(adjacency_matrix, dim=1).to_dense(), -1)
        norm_w[torch.isinf(norm_w)] = 0

        adjacency_matrix_norm_value = norm_w[row_indices] * adjacency_matrix_value
        self.adjacency_matrix_normed = torch.sparse_coo_tensor(torch.stack([row_indices, col_indices], dim=0), \
                                                                      adjacency_matrix_norm_value, (matrix_size, matrix_size)).to(self.device)


    def q_link_predict(self, item_degrees, top_rate, fast_weights):

        sorted_item_degrees = sorted(item_degrees.items(), key=lambda x: x[1])
        item_list_sorted, d_item = zip(*sorted_item_degrees)
        item_tail = torch.tensor(item_list_sorted).to(self.device)

        top_length = int(self.item_num * top_rate)
        item_top = torch.tensor(item_list_sorted[-top_length:]).to(self.device)


        encoder_0_weight = fast_weights[0]
        encoder_0_bias = fast_weights[1]
        encoder_2_weight = fast_weights[2]
        encoder_2_bias = fast_weights[3]

        top_item_feature = self.generator.embed_feature(item_top)
        tail_item_feature = self.generator.embed_feature(item_tail)

        top_item_hidden = torch.mm(top_item_feature, encoder_0_weight.t()) + encoder_0_bias
        top_item_embedded = torch.mm(top_item_hidden, encoder_2_weight.t()) + encoder_2_bias

        tail_item_hidden = torch.mm(tail_item_feature, encoder_0_weight.t()) + encoder_0_bias
        tail_item_embedded = torch.mm(tail_item_hidden, encoder_2_weight.t()) + encoder_2_bias

        
        i2i_score = torch.mm(tail_item_embedded, top_item_embedded.permute(1, 0))

        i2i_score_masked, indices = i2i_score.topk(self.link_topk, dim= -1)
        i2i_score_masked = i2i_score_masked.sigmoid()

        tail_item_degree = torch.sum(i2i_score_masked, dim=1)
        top_item_degree = torch.sum(i2i_score_masked, dim=0)
        tail_item_degree = torch.pow(tail_item_degree + 1, -1).unsqueeze(1).expand_as(i2i_score_masked).reshape(-1)
        top_item_degree = torch.pow(top_item_degree + 1, -1).unsqueeze(0).expand_as(i2i_score_masked).reshape(-1)


        tail_item_index = item_tail.unsqueeze(1).expand_as(i2i_score).gather(1, indices).reshape(-1)
        top_item_index = item_top.unsqueeze(0).expand_as(i2i_score).gather(1, indices).reshape(-1)
        enhanced_value = i2i_score_masked.reshape(-1)

        row_index = (tail_item_index+self.user_num).unsqueeze(0)
        colomn_index = (top_item_index+self.user_num).unsqueeze(0)
        joint_enhanced_value = enhanced_value * tail_item_degree
        
        return row_index, colomn_index, joint_enhanced_value


    def q_forward(self, user_id, pos_item, neg_item, fast_weights, inverse_pop = lambda x, k: k / (k + np.exp(x / k))):
        row_index, colomn_index, joint_enhanced_value = self.q_link_predict(self.item_degrees, self.top_rate, fast_weights)
        indice = torch.cat([row_index, colomn_index], dim=0).to(self.device)

        cur_embedding = torch.cat([self.user_id_Embeddings.weight, self.item_id_Embeddings.weight], dim=0)

        all_embeddings = [cur_embedding]
        enhance_weight = torch.from_numpy(inverse_pop(self.item_degree_numpy, self.convergence))
        enhance_weight = torch.cat([torch.zeros(self.user_num), enhance_weight], dim=-1).to(self.device).float()

        for i in range(self.L):
            cur_embedding_ori = torch.mm(self.adjacency_matrix_normed.to_dense(), cur_embedding)
            cur_embedding_enhanced = torch_sparse.spmm(indice, joint_enhanced_value, self.user_num + self.item_num, self.user_num + self.item_num, cur_embedding)
            cur_embedding = cur_embedding_ori + enhance_weight.unsqueeze(-1) * cur_embedding_enhanced
            all_embeddings.append(cur_embedding)

        all_embeddings = torch.stack(all_embeddings, dim=0)
        all_embeddings = torch.mean(all_embeddings, dim=0, keepdim=False)
        user_embeddings, item_embeddings = torch.split(all_embeddings, [self.user_num,self.item_num])

        user_embedded = user_embeddings[user_id]
        pos_item_embedded = item_embeddings[pos_item]
        neg_item_embedded = item_embeddings[neg_item]
        pos_score = torch.mul(user_embedded, pos_item_embedded).sum(dim=-1, keepdim=False)
        neg_score = torch.mul(user_embedded, neg_item_embedded).sum(dim=-1, keepdim=False)

        rec_loss = -(pos_score - neg_score).sigmoid().log().mean()
        
        return rec_loss

    def link_predict(self, item_degrees, top_rate):

        sorted_item_degrees = sorted(item_degrees.items(), key=lambda x: x[1])
        item_list_sorted, d_item = zip(*sorted_item_degrees)
        item_tail = torch.tensor(item_list_sorted).to(self.device)

        top_length = int(self.item_num * top_rate)
        item_top = torch.tensor(item_list_sorted[-top_length:]).to(self.device)


        top_item_embedded = self.generator.encode(item_top)
        tail_item_embedded = self.generator.encode(item_tail)
        
        i2i_score = torch.mm(tail_item_embedded, top_item_embedded.permute(1, 0))

        i2i_score_masked, indices = i2i_score.topk(self.link_topk, dim= -1)
        i2i_score_masked = i2i_score_masked.sigmoid()


        tail_item_degree = torch.sum(i2i_score_masked, dim=1)
        top_item_degree = torch.sum(i2i_score_masked, dim=0)
        tail_item_degree = torch.pow(tail_item_degree + 1, -1).unsqueeze(1).expand_as(i2i_score_masked).reshape(-1)
        top_item_degree = torch.pow(top_item_degree + 1, -1).unsqueeze(0).expand_as(i2i_score_masked).reshape(-1)


        tail_item_index = item_tail.unsqueeze(1).expand_as(i2i_score).gather(1, indices).reshape(-1)
        top_item_index = item_top.unsqueeze(0).expand_as(i2i_score).gather(1, indices).reshape(-1)
        enhanced_value = i2i_score_masked.reshape(-1)

        row_index = (tail_item_index+self.user_num).unsqueeze(0)
        colomn_index = (top_item_index+self.user_num).unsqueeze(0)
        joint_enhanced_value = enhanced_value * tail_item_degree
        
        return row_index, colomn_index, joint_enhanced_value

    # full item set
    def predict(self, user_id, inverse_pop = lambda x, k: k / (k + np.exp(x / k))):
        row_index, colomn_index, joint_enhanced_value = self.link_predict(self.item_degrees, self.top_rate)
        indice = torch.cat([row_index, colomn_index], dim=0).to(self.device)

        cur_embedding = torch.cat([self.user_id_Embeddings.weight, self.item_id_Embeddings.weight], dim=0)

        all_embeddings = [cur_embedding]

        enhance_weight = torch.from_numpy(inverse_pop(self.item_degree_numpy, self.convergence))
        enhance_weight = torch.cat([torch.zeros(self.user_num), enhance_weight], dim=-1).to(self.device).float()

        for i in range(self.L):
            cur_embedding_ori = torch.mm(self.adjacency_matrix_normed.to_dense(), cur_embedding)
            cur_embedding_enhanced = torch_sparse.spmm(indice, joint_enhanced_value, self.user_num + self.item_num, self.user_num + self.item_num, cur_embedding)
            cur_embedding = cur_embedding_ori + enhance_weight.unsqueeze(-1) * cur_embedding_enhanced
            all_embeddings.append(cur_embedding)

        all_embeddings = torch.stack(all_embeddings, dim=0)
        all_embeddings = torch.mean(all_embeddings, dim=0, keepdim=False)
        user_embeddings, item_embeddings = torch.split(all_embeddings, [self.user_num,self.item_num])

        user_embedded = user_embeddings[user_id]

        pos_item_embedded = item_embeddings

        score = torch.mm(user_embedded, pos_item_embedded.t())

        return score