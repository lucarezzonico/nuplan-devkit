import torch

def create_dict(*args):
    return dict({name:eval(name).shape for name in args})

from varname import nameof

if __name__ == '__main__':
    x = torch.zeros(2,2,3)
    y = torch.zeros(4,2,3)
    # # print_in_out([x,y])
    # for l in [x,y]:
    #     print(f'{l}')
    # feature_list = list(x).append(y)
    # print(nameof(feature_list[i]) for i in range(len(feature_list)))
    
    print(create_dict(nameof(x), nameof(y)))
    