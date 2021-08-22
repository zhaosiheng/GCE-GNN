import time
import argparse
import pickle
from model import *
from utils import *
import torch_xla.distributed.xla_multiprocessing as xmp

def init_seed(seed=None):
    if seed is None:
        seed = int(time.time() * 1000 // 1000)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='diginetica', help='diginetica/Nowplaying/Tmall')
parser.add_argument('--hiddenSize', type=int, default=100)
parser.add_argument('--epoch', type=int, default=20)
parser.add_argument('--activate', type=str, default='relu')
parser.add_argument('--n_sample_all', type=int, default=12)
parser.add_argument('--n_sample', type=int, default=12)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.001, help='learning rate.')
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay.')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay.')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty ')
parser.add_argument('--n_iter', type=int, default=1)                                    # [1, 2]
parser.add_argument('--dropout_gcn', type=float, default=0, help='Dropout rate.')       # [0, 0.2, 0.4, 0.6, 0.8]
parser.add_argument('--dropout_local', type=float, default=0, help='Dropout rate.')     # [0, 0.5]
parser.add_argument('--dropout_global', type=float, default=0.5, help='Dropout rate.')
parser.add_argument('--validation', action='store_true', help='validation')
parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=3)
parser.add_argument('--hop', type=int, default=5)
parser.add_argument('--long_edge_dropout', type=float, default=0.0)
parser.add_argument('--t', type=float, default=1.0)



opt = parser.parse_args()
SERIAL_EXEC = xmp.MpSerialExecutor()

def main():
    init_seed(2020)

    if opt.dataset == 'diginetica':
        num_node = 43098
        opt.n_iter = 1
        #opt.dropout_gcn = 0.3
        opt.dropout_local = 0.0
    elif opt.dataset == 'Nowplaying':
        num_node = 60417
        opt.n_iter = 1
        #opt.dropout_gcn = 0.0
        opt.dropout_local = 0.0
    elif opt.dataset == 'Tmall':
        num_node = 40728
        opt.n_iter = 1
        #opt.dropout_gcn = 0.4
        opt.dropout_local = 0.0
    else:
        num_node = 310
    
    def get_data(opt):
        train_data = pickle.load(open('datasets/' + opt.dataset + '/train.txt', 'rb'))
        if opt.validation:
            train_data, valid_data = split_validation(train_data, opt.valid_portion)
            test_data = valid_data
        else:
            test_data = pickle.load(open('datasets/' + opt.dataset + '/test.txt', 'rb'))
        adj = pickle.load(open('datasets/' + opt.dataset + '/adj_' + str(opt.n_sample_all) + '.pkl', 'rb'))
        num = pickle.load(open('datasets/' + opt.dataset + '/num_' + str(opt.n_sample_all) + '.pkl', 'rb'))
        train_data = Data(train_data, hop=opt.hop)
        test_data = Data(test_data, hop=opt.hop)
        adj, num = handle_adj(adj, num_node, opt.n_sample_all, num)
        return train_data, test_data, adj, num
    
    def map_fn(index, opt):
        init_seed(2020)
        train_data, test_data, adj, num = SERIAL_EXEC.run(lambda: get_data(opt))
        device = xm.xla_device()
        
        model = trans_to_cuda(CombineGraph(opt, num_node, adj, num, device), device)
        
        if xm.is_master_ordinal():  # Divergent CPU-only computation (no XLA tensors beyond this point!)
            print(opt)
            start = time.time()
            best_result = [0, 0, 0, 0]
            best_epoch = [0, 0, 0, 0]
            bad_counter = 0
            t0 = time.time()
        
        

        for epoch in range(opt.epoch):
            xm.master_print('-------------------------------------------------------')
            xm.master_print('epoch: ', epoch)
            train_test(model, train_data, device, index)
    ###
            
            xm.wait_device_ops()
            xm.master_print('start predicting: ', datetime.datetime.now())
            model.eval()



            if not xm.is_master_ordinal():
                xm.rendezvous('test_time')
            if xm.is_master_ordinal():
                test_loader = torch.utils.data.DataLoader(test_data, num_workers=4, batch_size=model.batch_size,
                                                          shuffle=False)
                result = []
                hit, mrr, hit_alias, mrr_alias = [], [], [], []
                for data in test_loader:
                    targets, scores = forward(model, data, device)
                    sub_scores = scores.topk(20)[1]
                    sub_scores_alias = scores.topk(10)[1]
                    sub_scores = trans_to_cpu(sub_scores).detach().numpy()
                    sub_scores_alias = trans_to_cpu(sub_scores_alias).detach().numpy()
                    targets = targets.numpy()
                    for score, target, mask in zip(sub_scores, targets, test_data.mask):
                        #@20
                        hit.append(np.isin(target - 1, score))
                        if len(np.where(score == target - 1)[0]) == 0:
                            mrr.append(0)
                        else:
                            mrr.append(1 / (np.where(score == target - 1)[0][0] + 1))

                    for score, target, mask in zip(sub_scores_alias, targets, test_data.mask):
                        #@10
                        hit_alias.append(np.isin(target - 1, score))
                        if len(np.where(score == target - 1)[0]) == 0:
                            mrr_alias.append(0)
                        else:
                            mrr_alias.append(1 / (np.where(score == target - 1)[0][0] + 1))


                hit = np.mean(hit) * 100
                mrr = np.mean(mrr) * 100

                hit_alias = np.mean(hit_alias) * 100
                mrr_alias = np.mean(mrr_alias) * 100
        ###
                
                flag = 0
                if hit >= best_result[0]:
                    best_result[0] = hit
                    best_epoch[0] = epoch
                    flag = 1
                if mrr >= best_result[1]:
                    best_result[1] = mrr
                    best_epoch[1] = epoch
                    flag = 1
                if hit_alias >= best_result[2]:
                    best_result[2] = hit_alias
                    best_epoch[2] = epoch
                    flag = 1
                if mrr_alias >= best_result[3]:
                    best_result[3] = mrr_alias
                    best_epoch[3] = epoch
                    flag = 1
                print('Current Result:')
                print('\tRecall@20:\t%.4f\tMMR@20:\t%.4f\tRecall@10:\t%.4f\tMMR@10:\t%.4f' % (hit, mrr, hit_alias, mrr_alias))
                print('Best Result:')
                print('\tRecall@20:\t%.4f\tMMR@20:\t%.4f\tRecall@10:\t%.4f\tMMR@10:\t%.4f\tEpoch:\t%d,\t%d,\t%d,\t%d' % (
                    best_result[0], best_result[1], best_result[2], best_result[3], best_epoch[0], best_epoch[1], best_epoch[2], best_epoch[3]))
                t1 = time.time()
                print('time', t1 - t0)
                bad_counter += 1 - flag
                if bad_counter >= opt.patience:
                    break
                xm.rendezvous('test_time')
        
        if xm.is_master_ordinal():
            print('-------------------------------------------------------')
            end = time.time()
            print("Run time: %f s" % (end - start))

        xm.rendezvous('finish')
    flags = opt
    xmp.spawn(map_fn, args=(flags,), nprocs=8, start_method='fork')
        

    #map_fn(0, opt)
    
if __name__ == '__main__':
    main()
