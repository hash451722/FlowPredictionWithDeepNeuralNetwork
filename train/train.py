import pathlib

import numpy as np

import torch
import torchvision.transforms as transforms

import dataset
import network
import visual



def train_network():

    # User settings
    epochs = 3

    # learning rate
    learning_rate = 3e-4   #0.0006

    # GPU
    use_cuda = True

    batch_size = 10


    # Define what device we are using
    # print("CUDA Available: ",torch.cuda.is_available())
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")


    # 1. Load training , validation and test datasets
    # transform = transforms.Compose([transforms.ToTensor()])
    # transform = transforms.Compose([])

    # 全データのデータセット
    ds = dataset.FpdnnDataset(transform=None)

    # Randomly split the dataset (train, valid, test)
    train_frac, valid_frac, test_frac = (0.7, 0.2, 0.1)
    n_ds = len(ds)
    n_train = int( len(ds) * train_frac / (train_frac+valid_frac+test_frac) )
    n_valid = int( len(ds) * valid_frac / (train_frac+valid_frac+test_frac) )
    n_test = n_ds - n_train - n_valid
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(ds, [n_train, n_valid, n_test])

    trainloader = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=2)

    validloader = torch.utils.data.DataLoader(valid_dataset,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            num_workers=2)

    testloader = torch.utils.data.DataLoader(test_dataset,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            num_workers=2)




    # Instantiate a neural network model 
    # 2. Define a Neural Network
    net = network.select_net("FpDnn").to(device)


    # 3. Define a Loss function and optimizer
    criterion = torch.nn.L1Loss().to(device)
    optimizer = torch.optim.Adam(net.parameters(),
                                 lr=learning_rate,
                                 betas=(0.5, 0.999),
                                 weight_decay=0.0)



    # 4. Training and validation the network
    for epoch in range(epochs):

        # Training
        net.train()
        running_loss = 0.0
        for i, (inputs, targets) in enumerate(trainloader):
            # print(i, inputs.shape, targets.shape)

            inputs = inputs.to(device)
            targets = targets.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        loss_train = running_loss / len(train_dataset)


        # Vlidation
        net.eval()
        running_loss  = 0.0
        with torch.no_grad():
            for inputs, targets in validloader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)
                running_loss  += loss.item()

        loss_valid = running_loss / len(valid_dataset)


        # Output to terminal
        print("epoch {} / {}".format((epoch+1),epochs))
        print(loss_train, loss_valid)

    print("=== Finished Training' ===")


    # 5. Save the model
    dir_path = pathlib.Path(__file__).parent 
    pt_path = dir_path.joinpath("fpdnn.pt")
    torch.save(net.state_dict(), pt_path)

    # Exporting the model to ONNX
    dummy_input = torch.randn(1, 3, 128, 128, device=device)
    onnx_path = dir_path.joinpath("fpdnn.onnx")
    torch.onnx.export(net, dummy_input , onnx_path)



    # 6. visualize
    # inputs, targets = next(iter(trainloader))
    # input =  inputs[0].unsqueeze(0)
    # target = targets[0].unsqueeze(0)

    # net.eval()
    # with torch.no_grad():
    #     input = input.to(device)
    #     output = net(input)

    # visual.draw_data(torch.vstack((input[0],
    #                                target[0],
    #                                output[0])))




if __name__ == "__main__":
    train_network()
