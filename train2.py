import torch
from torch import optim
import torch.autograd
from torch.utils.data import DataLoader

import numpy as np
from collections import defaultdict

from model2 import Model

import load_data2 as load_data

from tqdm import tqdm
import metric
import argparse
import os

def test(Data, opt):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    test_dataset = Data.test_dataset
    test_loader = DataLoader(
        test_dataset, shuffle=False, batch_size=opt.batch_size, collate_fn=None)
    
    best_checkpoint = torch.load(os.path.join(os.path.dirname(__file__), 'model', opt.model), map_location=device)

    opt = best_checkpoint['opt']
    model = Model(Data, opt, device)
    model.load_state_dict(best_checkpoint['sd'])
    model = model.to(device)

    model.eval()
    user_historical_mask = Data.user_historical_mask.to(device)

    NDCG = defaultdict(list)
    RECALL = defaultdict(list)

    head_NDCG = defaultdict(list)
    head_RECALL = defaultdict(list)
    tail_NDCG = defaultdict(list)
    tail_RECALL = defaultdict(list)

    with torch.no_grad():
        with tqdm(total=len(test_loader), desc="predicting") as pbar:
            for _, (user_id, observed_item) in enumerate(test_loader):
                user_id = user_id.to(device)

                score = model.predict(user_id)
                score = torch.mul(user_historical_mask[user_id], score).cpu().detach().numpy()
                ground_truth = observed_item.detach().numpy()

                for K in opt.K_list:
                    metrics = metric.ranking_measure_testset(score, ground_truth, K, list(Data.testset_item.keys()))
                    metrics_head_tail = metric.ranking_measure_degree_testset(score, ground_truth, K, Data.item_degrees, opt.seperate_rate, list(Data.testset_item.keys()))
                    NDCG[K].append(metrics['ndcg'])
                    RECALL[K].append(metrics['recall'])
                
                    head_NDCG[K].append(metrics_head_tail['head']['ndcg'])
                    head_RECALL[K].append(metrics_head_tail['head']['recall'])
                    tail_NDCG[K].append(metrics_head_tail['tail']['ndcg'])
                    tail_RECALL[K].append(metrics_head_tail['tail']['recall'])

                pbar.update(1)

        print(opt)
        print(model.name)
        for K in opt.K_list:
            print("NDCG@{}: {}".format(K, np.mean(NDCG[K])))
            print("RECALL@{}: {}".format(K, np.mean(RECALL[K])))
            print('\r\r')
            print("head_NDCG@{}: {}".format(K, np.mean(head_NDCG[K])))
            print("head_RECALL@{}: {}".format(K, np.mean(head_RECALL[K])))
            print('\r\r')
            print("tail_NDCG@{}: {}".format(K, np.mean(tail_NDCG[K])))
            print("tail_RECALL@{}: {}".format(K, np.mean(tail_RECALL[K])))
            print('\r\r')

def generate_item_pair(Data, opt, device):
    # [...userindex, ...itemindex]. 
    # Can be considered as zip(userindex, itemindex) for all interactions.
    index = [Data.interact_train['userid'].tolist(), Data.interact_train['itemid'].tolist()]
    value = [1.0] * len(Data.interact_train)

    # R, which is of shape (user_num, item_num).
    # Each row is a user's interaction with items.
    # Stored in sparse tensor.
    interact_matrix = torch.sparse_coo_tensor(index, value, (Data.user_num, Data.item_num)).to(device)

    # R^T * R
    rtr = torch.sparse.mm(interact_matrix.t().to_dense(), interact_matrix.to_dense())
    rtr = rtr.to_sparse()

    def sparse_min(A):
        A = A.coalesce()
        A_values = A.values()
        A_indices = A.indices()
        A_values = torch.where(A_values > 1, A_values.new_ones(A_values.shape), A_values)
        return torch.sparse_coo_tensor(A_indices, A_values, A.shape).to(A.device)

    # S'
    s_prime = sparse_min(rtr)

    def get_mask(item_num, rate=0.2):
        zeros_num = int(item_num * rate)
        new_array = np.ones((item_num, item_num))
        for row in new_array:
            row[:zeros_num] = 0
            np.random.shuffle(row)
        re_array = torch.from_numpy(new_array).to_sparse().to(device)
        return re_array

    # M
    mask = get_mask(Data.item_num)
    
    # inplace operation
    # S'_mask = S' .* M .* M^T
    s_prime.mul_(mask).mul_(mask.t())

    s_prime_mask = s_prime.coalesce()

    item1 = s_prime_mask.indices()[0].tolist()
    item2 = s_prime_mask.indices()[1].tolist()

    # we retrieve (i, j) where S'_mask_{i, j} = 1
    item_pair = list(zip(item1, item2))

    return item_pair

def one_train(Data, opt):
    print(opt)
    print('Building dataloader >>>>>>>>>>>>>>>>>>>')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(device)

    print("building model >>>>>>>>>>>>>>>")
    model = Model(Data, opt, device)

    print('Building optimizers >>>>>>>')
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    print('Start training...')
    start_epoch = 0
    # directory = directory_name_generate(model, opt, "no early stop")
    model = model.to(device)

    item_pair = generate_item_pair(Data, opt, device)
    support_loader = DataLoader(item_pair, shuffle=True, batch_size=opt.batch_size, collate_fn=None)

    recalls = []
    for epoch in range(start_epoch, opt.epoch):
        model.train()

        train_loader = DataLoader(Data.train_dataset, shuffle=True, batch_size=opt.batch_size, collate_fn=None)
        support_iter = iter(support_loader)

        train_loss = 0
        with tqdm(total=len(train_loader), desc="epoch"+str(epoch)) as pbar:
            for index, (user_id, observed_item, unobserved_item) in enumerate(train_loader):
                user_id = user_id.to(device)
                observed_item = observed_item.to(device)
                unobserved_item = unobserved_item.to(device)

                item1, item2 = next(support_iter)
                item1 = item1.to(device)
                item2 = item2.to(device)

                # F = L_GL + lambda * L_PCL. Refer to equation (15)
                F = model.gl_loss(item1, item2) + opt.reg_lambda * model.pcl_loss(item1)

                # theta
                theta = list(model.generator.encoder.state_dict().values())

                # F'. Refer to equation (17)
                grad_F = torch.autograd.grad(F, model.generator.encoder.parameters(), create_graph=True, allow_unused=True)
                
                new_theta = []
                for i, weight in enumerate(theta):
                    new_theta.append(weight - opt.local_lr * grad_F[i])
                # J
                J = model.q_forward(user_id, observed_item, unobserved_item, new_theta)

                # equation (17)
                loss = J + opt.beta * F

                train_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pbar.update(1)

        print('epoch: ', epoch, 'loss: ', train_loss)

        model.eval()
        with torch.no_grad():
            val_loader = DataLoader(Data.val_dataset, shuffle=False, batch_size=opt.batch_size, collate_fn=None)
            user_historical_mask = Data.user_historical_mask.to(device)
            RECALL20 = []

            with tqdm(total=len(val_loader), desc="validating") as pbar:
                for _, (user_id, observed_item) in enumerate(val_loader):
                    user_id = user_id.to(device)
                    # observed_item (uuu * item_num)
                    # uuu * item_num
                    score = model.predict(user_id)
                    score = torch.mul(user_historical_mask[user_id], score).cpu().detach().numpy()
                    ground_truth = observed_item.detach().numpy()

                    for K in opt.K_list:
                        metrics = metric.ranking_measure_testset(score, ground_truth, K, list(Data.valset_item.keys()))
                        RECALL20.append(metrics['recall'])

                    pbar.update(1)

            recall20 = np.mean(RECALL20)
            print('val recall 20: ', recall20)

            # stop when recall@20 is not increasing for 3 epochs
            if len(recalls) > 2 and recall20 < recalls[-1] and recall20 < recalls[-2] and recall20 < recalls[-3]:
                print('early stop: recall@20 is not increasing for 3 epochs')
                break
            
            if len(recalls) == 0:
                recalls.append(recall20)
            else:
                recalls.append(max(recalls[-1], recall20))

            torch.save({
                'sd': model.state_dict(),
                'opt':opt,
            }, os.path.join(os.path.dirname(__file__), 'model', opt.output))
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_name", type=str, default='book_crossing')
    parser.add_argument("--social_data", type=bool, default=False)
    # test_set/cv/split
    parser.add_argument("--load_mode", type=str, default='test_set')

    parser.add_argument("--implcit_bottom", type=int, default=None)
    parser.add_argument("--cross_validate", type=int, default=None)
    parser.add_argument("--split", type=float, default=None)
    parser.add_argument("--user_fre_threshold", type=int, default=None)
    parser.add_argument("--item_fre_threshold", type=int, default=None)

    parser.add_argument("--loadFilename", type=str, default=None)

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epoch", type=int, default=300)

    parser.add_argument("--embedding_size", type=int, default=8)
    parser.add_argument("--id_embedding_size", type=int, default=64)
    parser.add_argument("--dense_embedding_dim", type=int, default=16)

    parser.add_argument("--L", type=int, default=3)
    
    parser.add_argument("--link_topk", type=int, default=10)

    parser.add_argument("--reg_lambda", type=float, default=0.02)
    parser.add_argument("--top_rate", type=float, default=0.1)
    parser.add_argument("--convergence", type=float, default=40)
    parser.add_argument("--seperate_rate", type=float, default=0.2)
    parser.add_argument("--local_lr", type=float, default=0.01)
    parser.add_argument("--beta", type=float, default=0.1)

    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.01)

    parser.add_argument("--K_list", type=int, nargs='+', default=[10, 20])

    parser.add_argument("--output", type=str, default="model.tar")
    parser.add_argument("--model", type=str, default="model.tar")

    opt = parser.parse_args()

    Data = load_data.Data(opt)
    one_train(Data, opt)
    test(Data, opt)
