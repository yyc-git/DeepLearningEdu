from operator import truth
import os
import numpy as np
import pyexr

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio

import dataset
import net

def BMFRGammaCorrection(img):
    if isinstance(img, np.ndarray):
        return np.clip(np.power(np.maximum(img, 0.0), 0.454545), 0.0, 1.0)
    elif isinstance(img, torch.Tensor):
        return torch.pow(torch.clamp(img, min=0.0, max=1.0), 0.454545)

def ComputeMetrics(truth_img, test_img):    
    truth_img = BMFRGammaCorrection(truth_img)
    test_img  = BMFRGammaCorrection(test_img)

    # print(truth_img.shape)
    # print(test_img.shape)

    # im1 = truth.astype(float_type, copy=False)
    # # im2 = im2.astype(float_type, copy=False)

    # print(np.asarray(im1.shape))

    # truth_img = np.squeeze(truth_img)
    # test_img = np.squeeze(test_img)
    
    # SSIM = structural_similarity(truth_img, test_img, multichannel=True)
    # SSIM = structural_similarity(truth_img, test_img, channel_axis=2, data_range=float,
    
    # gaussian_weights = True, sigma = 1.5, use_sample_covariance = False
    # )
    SSIM = 0.0

    PSNR = peak_signal_noise_ratio(truth_img, test_img)
    return SSIM, PSNR

class SMAPELoss(nn.Module):
    def __init__(self, eps=0.01):
        super(SMAPELoss, self).__init__()
        self.eps = eps
    
    def forward(self, outputs, targets):
        denominator = outputs
        loss = torch.mean(torch.abs(outputs - targets) / (denominator.abs() + targets.abs() + self.eps))
        return loss
    


def train(model, device, dataloader, optimizer, epoch, writer):
    model.train()
    print("aaa")
    losses = []
    criterion = SMAPELoss().to(device)

    for (inputs, targets) in dataloader:
        print("bbb")
        optimizer.zero_grad(set_to_none=True)
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()

        print("ccc")
        optimizer.step()
        print("ddd")
        losses.append(loss.item())
        # TODO remove
        break
        
    print("333")
    writer.add_scalar("Loss/total_train", np.mean(losses), epoch)
    print(np.mean(losses))

def Inference(model, device, dataset, dataloader, saving_root=""):
    model.eval()
    SSIMs = []
    PSNRs = []
    with torch.no_grad():
        for img_idx, (inputs_crops, targets_crops) in enumerate(dataloader):
            inputs = inputs_crops.to(device, non_blocking=True)
            targets = targets_crops.to(device, non_blocking=True)
            outputs = model(inputs).detach()
            
            if dataset.use_val:            
                output = outputs.cpu().numpy()
                target = targets.cpu().numpy()
                # print(output)
                # print(target)
                for i in range(output.shape[0]):
                    if np.sum(target[i]) == 0.0:
                        continue
                    SSIM, PSNR = ComputeMetrics(target[i].transpose((1, 2, 0)), output[i].transpose((1, 2, 0)))
                    SSIMs.append(SSIM)
                    PSNRs.append(PSNR)
            
            elif dataset.use_test:
                # batch size in test is 1
                output = outputs.cpu().numpy()[0].transpose((1, 2, 0)) # BMFR
                target = targets.cpu().numpy()[0].transpose((1, 2, 0))
                SSIM, PSNR = ComputeMetrics(target, output)
                SSIMs.append(SSIM)
                PSNRs.append(PSNR)
                
            if dataset.use_test:
                pyexr.write(os.path.join(saving_root, str(img_idx)+".exr"), output)
            # if dataset.use_val:            
            #     output = outputs.cpu().numpy()[0].transpose((1, 2, 0)) # BMFR
            #     pyexr.write(os.path.join(saving_root, str(img_idx)+".exr"), output)
            
    if dataset.use_val:
        print("Validation:")
    elif dataset.use_test:
        print("Test:")
    SSIM_mean = np.mean(SSIMs)
    PSNR_mean = np.mean(PSNRs)
    print("mean SSIM:", SSIM_mean)
    print("mean PSNR:", PSNR_mean)
    SSIMs.append("mean: "+str(SSIM_mean))
    PSNRs.append("mean: "+str(PSNR_mean))
    if dataset.use_test:
    # if dataset.use_val:            
        np.savetxt(os.path.join(saving_root, "ssim.txt"), SSIMs, fmt="%s")
        np.savetxt(os.path.join(saving_root, "psnr.txt"), PSNRs, fmt="%s")
            
    return SSIM_mean, PSNR_mean


if __name__ == "__main__":

    # torch.cuda.set_device(1)
    # torch.backends.cudnn.deterministic = True  # same result for cpu and gpu
    # torch.backends.cudnn.benchmark = False # key in here: Should be False. Ture will make the training process unstable
    # device = torch.device("cuda")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    learning_rate = 1e-3 # BMFR
    # epochs = 500
    # epochs = 10
    # epochs = 3
    epochs = 1
    # epoch_test = 25
    # epoch_test = 3
    epoch_test = 1
    batch_size = 64

    database = dataset.DataBase()
    dataset_train = dataset.BMFRFullResAlDataset(database, use_train=True)
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    dataset_val = dataset.BMFRFullResAlDataset(database, use_val=True)
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    dataset_test = dataset.BMFRFullResAlDataset(database, use_test=True)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    timestamp = "open-source-test"
    # episode_name = "sponza"
    episode_name = "classroom"
    tensorboard_saving_path = os.path.join("runs", timestamp, episode_name)
    test_saving_root = os.path.join("results", timestamp, episode_name)
    model_saving_path = os.path.join("checkpoints", timestamp, episode_name)
    for folder in [tensorboard_saving_path, model_saving_path, test_saving_root]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    model = net.repWeightSharingKPNet(device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    writer = SummaryWriter(tensorboard_saving_path)



    for epoch in range(epochs):
        print(epoch)
        # Train
        train(model, device, dataloader_train, optimizer, epoch, writer)
        # Evaluation
        if (epoch+1) % epoch_test == 0:
            # print("begin val")
            # _SSIM_val, _PSNR_val = Inference(model, device, dataset_val, dataloader_val)
            # # _SSIM_val, _PSNR_val = Inference(model, device, dataset_val, dataloader_val, test_saving_root)
            # writer.add_scalar("SSIM-val", _SSIM_val, epoch)
            # writer.add_scalar("PSNR-val", _PSNR_val, epoch)

            # if epoch > epochs * 0.8:
            if True:
                print("begin test")
                _SSIM_test, _PSNR_test = Inference(model, device, dataset_test, dataloader_test, test_saving_root)
                writer.add_scalar("SSIM-test", _SSIM_test, epoch)
                writer.add_scalar("PSNR-test", _PSNR_test, epoch)                
            
            torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, os.path.join(model_saving_path, "model.pt"))

    writer.flush()
    writer.close()