import pathlib

import numpy as np
import torch
import torchvision.transforms as transforms


class FpdnnDataset(torch.utils.data.Dataset):
    def __init__(self, transform=None) -> None:
        ''' mode : train  or  test or  valid '''
        self.transform = transform

        train_data_path = pathlib.Path(__file__).parent.parent.joinpath("data", "train_data")
        self.train_data = list(train_data_path.glob("*.npy"))


    def __getitem__(self, index:int) -> tuple:
        ar = np.load(self.train_data[index])
        inputs, targets = np.vsplit(ar, 2)

        # print(type(input_data))

        if self.transform is not None:
            inputs = self.transform(inputs)
            targets = self.transform(targets)

        x = torch.from_numpy(inputs.astype(np.float32)).clone()
        y = torch.from_numpy(targets.astype(np.float32)).clone()
        return x, y

    def __len__(self) -> int:
        return len(self.train_data)



def test():
    transform = transforms.Compose([transforms.ToTensor()])

    dataset = FpdnnDataset(mode="train", transform=None)

    trainloader = torch.utils.data.DataLoader(dataset,
                                              batch_size = 4,
                                              shuffle = True,
                                              num_workers = 2)

    for i, (inputs, targets) in enumerate(trainloader):
        print(i, inputs.shape, targets.shape)


if __name__ == "__main__":
    test()
