from data_provider.data_loader import Dataset_Custom, Dataset_Pred, My_pred
from torch.utils.data import DataLoader

def data_provider(args, flag, data_dict, industry):
    type_dict = {
    'custom': Dataset_Custom,
    }

    Data = type_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        Data = My_pred
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size

    data_set = Data(
        data_dict=data_dict,
        industry=industry,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
)
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        drop_last=drop_last)
    return data_set, data_loader
