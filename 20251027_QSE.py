import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from model20251027 import QSE
import time
import torch.optim as optim
import os                    
import torch
from torch import nn
#from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from monai.visualize import plot_2d_or_3d_image
from dataset20251027 import get_5EchoT2w_dataset_2d_3map
from monai.data import DataLoader, pad_list_data_collate, ThreadDataLoader
from utils import PathEncoder, initialize_weights, skimage_ssim, save_sourcecode
from pathlib import Path
from skimage.metrics import peak_signal_noise_ratio as psnr
import random


def gen_mask_gaussian2(kspace_shape, dividend=13, accel_factor=2, seed=10):
    # inspired by https://github.com/facebookresearch/fastMRI/blob/master/common/subsample.py
    shape = kspace_shape
    num_cols = shape[-2]

    center_fraction = (dividend // accel_factor) / 100      #af=2,32 af=3,48
    acceleration = accel_factor

    # Create the mask
    num_low_freqs = int(round(num_cols * center_fraction))
    _, remainder = divmod(num_low_freqs, 2)
    num_low_freqs = num_low_freqs if (remainder == 0) else (num_low_freqs + 1)
    num_high_freqs = int(round(num_cols / acceleration - num_low_freqs))


    # num_low_freqs = 32
    # num_high_freqs = 32

    high_idx_over = np.round(
        np.random.default_rng(seed).normal(
            num_cols // 2,
            num_cols // 4,
            int(num_high_freqs * 2.5)
        )
    ).astype(int)

    high_idx_over = np.array(list((set(high_idx_over.squeeze().tolist()))))   #set 删除重复数据

    high_idx_over = np.delete(
        high_idx_over, np.where(high_idx_over < 0)
    )

    high_idx_over = np.delete(
        high_idx_over, np.where(high_idx_over >= num_cols)
    )

    low = high_idx_over >= (num_cols // 2 - num_low_freqs // 2)
    high = high_idx_over < (num_cols // 2 + num_low_freqs // 2)
    high_idx_over = np.delete(
        high_idx_over, np.where(low & high)
    )

    if len(high_idx_over) <= num_high_freqs:
        high_idx_tmp = high_idx_over
    else:
        len_over = len(high_idx_over) - num_high_freqs
        random.seed(seed)
        idx_over = random.sample(range(1, len(high_idx_over)), len_over)
        high_idx_tmp = np.delete(high_idx_over, idx_over)

    high_idx = high_idx_tmp
    mask = np.zeros(num_cols)
    pad = (num_cols - num_low_freqs + 1) // 2
    mask[pad: pad + num_low_freqs] = True
    mask[high_idx] = True

    # Reshape the mask
    mask_shape = [1 for _ in shape]
    mask_shape[-1] = num_cols
    mask = mask.reshape(*mask_shape)
    mask = np.reshape(mask, [mask.shape[1], mask.shape[0]])
    fourier_mask = np.repeat(mask.astype(np.float64), shape[1], axis=1)
    
    return fourier_mask

ETL=5
mask1=torch.rand(ETL,256,256)
for m in range(ETL):
    seednum=np.random.randint(0,1001)
    mask01=gen_mask_gaussian2(kspace_shape=[256, 256], accel_factor=3, seed=seednum)
    mask01=torch.tensor(mask01)
    mask1[m,:,:]=mask01[:,:]

def normalize_to_range(img, min_val, max_val):
    """将图像数据归一化到 [0, 1] 范围内"""
    return (img - min_val) / (max_val - min_val)

device = 'cuda'
batch_size = 2
num_workers = 4

cache_train_ds=get_5EchoT2w_dataset_2d_3map(phase="train", device=device)
if device == 'cpu':
    train_loader = DataLoader(cache_train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available())
else:
    train_loader = ThreadDataLoader(cache_train_ds, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=pad_list_data_collate)
cache_val_ds=get_5EchoT2w_dataset_2d_3map(phase="val", prob=0, device=device)
if device == 'cpu':
    val_loader = DataLoader(cache_val_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available())
else:
    val_loader = ThreadDataLoader(cache_val_ds, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=pad_list_data_collate)

epoch_num = 5000 #itration number 

exp_name = (
    f'QSE3'
    f'_{time.strftime("%M%S")}'
)
output_dir = Path.home() / "QSE_result" / exp_name
output_dir.mkdir(parents=True, exist_ok=True)
save_sourcecode(code_rootdir=Path(__file__).parent, out_dir=output_dir)
writer = SummaryWriter(output_dir)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = QSE(useBN=True)
initialize_weights(model)
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)
model = model.to(device)
total_loss=0

criterion1 = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, weight_decay=0.0)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(epoch_num*1.2))
total_loss=0
tic = time.time()
best_ssim_all = -1
best_psnr_all = -1

for epoch in range(epoch_num):   #set to 0 for no running the training
    train_epoch_loss = 0
    train_step = 0
    model.train()
    loss_batch = []
    time_start=time.time()
    i=0    
    for data in train_loader:
        i = i+1
        train_step = train_step+1
        target_t2map, target_pdmap, target_t1map = (data["t2m"], data["pdm"], data["t1m"])
        mask = (target_t2map > 0).float()
        pd=target_pdmap[:,0,:,:]
        t1=target_t1map[:,0,:,:]
        t2=target_t2map[:,0,:,:]
        pd = torch.abs(pd)
        # 将 t2map_slice 和 t1map_slice 中的0替换为一个极小的正数
        epsilon = 1e-8
        t2 = t2.clamp(min=epsilon)
        t1 = t1.clamp(min=epsilon)

        ETL=5
        mean=0
        sigma=0.05
        T2w_mean_i=torch.rand(ETL,t2.shape[-3],t2.shape[-2],t2.shape[-1])
        T2w_mean_i=T2w_mean_i.to('cuda')
        real=torch.rand(ETL,t2.shape[-3],t2.shape[-2],t2.shape[-1])
        imag=torch.rand(ETL,t2.shape[-3],t2.shape[-2],t2.shape[-1])
        T2w_mean_k = torch.complex(real,imag)
        T2w_mean_k=T2w_mean_k.to('cuda')
        T2w_mean_k1 = torch.complex(real,imag)
        T2w_mean_k1=T2w_mean_k1.to('cuda')
        real1=torch.rand(1,t2.shape[-3],t2.shape[-2],t2.shape[-1])
        imag1=torch.rand(1,t2.shape[-3],t2.shape[-2],t2.shape[-1])
        T2w_k = torch.complex(real1,imag1)
        T2w_k=T2w_k.to('cuda')
        TR=6000
        esp=50

        #mask1=torch.rand(ETL,256,256)
        for m in range(ETL):
            T2w_mean_i[m,:,:,:] = torch.abs(pd * torch.exp(-esp*(m+1) / t2) * (1 -2*torch.exp(-(TR-esp*(m+1)/2) / t1)+ torch.exp(-TR / t1)))
        #T2w_mean_imax=torch.max(T2w_mean_i)
        
        T2w_mean_imax=torch.amax(T2w_mean_i,dim=(0,2,3))
        T2w_mean_imax=T2w_mean_imax+epsilon
        T2w_mean_imax_expand=T2w_mean_imax.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        T2w_mean_i=T2w_mean_i/T2w_mean_imax_expand
        T2w_mean_k1=torch.fft.fftshift(torch.fft.fft2((torch.fft.fftshift(T2w_mean_i, dim=(-2,-1))), dim=(-2,-1)), dim=(-2,-1))
        for m in range(T2w_mean_k1.shape[1]):
            temp=T2w_mean_k1[:,m,:,:]
            data_std=torch.std(temp)
            T2w_mean_k[:,m,:,:] =T2w_mean_k1[:,m,:,:]+(torch.complex(torch.normal(mean, sigma*data_std, temp.shape),torch.normal(mean, sigma*data_std, temp.shape))).to('cuda')
        
        mask11=mask1.unsqueeze(1)
        mask11=mask11.repeat(1,T2w_mean_k.shape[1], 1, 1) 
        mask11 = mask11.to(device)
        mask11 = mask11.to(torch.float32)
        #print(mask11[0,0,:,1])

        T2w_mean_k_mask=T2w_mean_k*mask11
        #print(T2w_mean_k_mask[0,0,:,1])
        
        T2w=torch.abs(torch.fft.fftshift(torch.fft.ifft2((torch.fft.fftshift(T2w_mean_k_mask, dim=(-2,-1))), dim=(-2,-1)), dim=(-2,-1)))    
        T2w=T2w.permute(1,0,2,3)
        T2w_ref=torch.abs(torch.fft.fftshift(torch.fft.ifft2((torch.fft.fftshift(T2w_mean_k, dim=(-2,-1))), dim=(-2,-1)), dim=(-2,-1))) 
        T2w_ref=T2w_ref.permute(1,0,2,3)
        T2w_mean_k_mask_input=T2w_mean_k_mask.permute(1,0,2,3)
        mask_input=mask11.permute(1,0,2,3)

        t2m, image, kspace = model(T2w_mean_k_mask_input.to(device),mask_input.to(device))
        t2m = torch.relu(t2m)
        image = torch.relu(image)
        
        kspace_mask=kspace*mask_input

    

        T2w_mean_iCAL=torch.rand(ETL,t2.shape[-3],1,t2.shape[-2],t2.shape[-1])
        T2w_mean_iCAL=T2w_mean_iCAL.to('cuda')
        realCAL=torch.rand(ETL,t2.shape[-3],1,t2.shape[-2],t2.shape[-1])
        imagCAL=torch.rand(ETL,t2.shape[-3],1,t2.shape[-2],t2.shape[-1])
        T2w_mean_kCAL = torch.complex(realCAL,imagCAL)
        T2w_mean_kCAL=T2w_mean_kCAL.to('cuda')
    
        T2w_mean_iCAL[0,:,:,:,:] = image[:,:,0,:,:]
        T2w_mean_kCAL[0,:,:,:,:] =torch.fft.fftshift(torch.fft.fft2((torch.fft.fftshift(T2w_mean_iCAL[0,:,:,:,:], dim=(-2,-1))), dim=(-2,-1)), dim=(-2,-1))
        
        M0_updata=image[:,:,0,:,:]
        M0_show=M0_updata.clone()
        M0_show[mask == 0] = 0  
            
        for m in range(ETL-1):
            T2w_mean_iCAL[m+1,:,:,:,:] = torch.abs(M0_updata * torch.exp(-esp*(m+1) / t2m))
            T2w_mean_kCAL[m+1,:,:,:,:] =torch.fft.fftshift(torch.fft.fft2((torch.fft.fftshift(T2w_mean_iCAL[m+1,:,:,:,:], dim=(-2,-1))), dim=(-2,-1)), dim=(-2,-1))
        T2w_mean_kCALc=T2w_mean_kCAL[:,:,0,:,:]
        
        image_pre1=T2w_mean_iCAL[0,:,:,:,:]
        image_pre2=T2w_mean_iCAL[1,:,:,:,:]
        image_pre3=T2w_mean_iCAL[2,:,:,:,:]
        image_pre4=T2w_mean_iCAL[3,:,:,:,:]
        image_pre5=T2w_mean_iCAL[4,:,:,:,:]
        image_pre1[mask==0]=0
        image_pre2[mask==0]=0
        image_pre3[mask==0]=0
        image_pre4[mask==0]=0
        image_pre5[mask==0]=0
        
        x_t2w1=image[:,:,0,:,:].clone()
        x_t2w2=image[:,:,1,:,:].clone()
        x_t2w3=image[:,:,2,:,:].clone()
        x_t2w4=image[:,:,3,:,:].clone()
        x_t2w5=image[:,:,4,:,:].clone()
        x_t2w1[mask==0]=0
        x_t2w2[mask==0]=0
        x_t2w3[mask==0]=0
        x_t2w4[mask==0]=0
        x_t2w5[mask==0]=0
        
        mask111=mask11[0,:,:,:]
        mask112=mask11[1,:,:,:]
        mask113=mask11[2,:,:,:]
        mask114=mask11[3,:,:,:]
        mask115=mask11[4,:,:,:]
        T2w_k_cal01=T2w_mean_k[0,:,:,:]
        T2w_k_cal02=T2w_mean_k[1,:,:,:]
        T2w_k_cal03=T2w_mean_k[2,:,:,:]
        T2w_k_cal04=T2w_mean_k[3,:,:,:]
        T2w_k_cal05=T2w_mean_k[4,:,:,:]
        T2w_k_pre01=T2w_mean_kCALc[0,:,:,:]
        T2w_k_pre02=T2w_mean_kCALc[1,:,:,:]
        T2w_k_pre03=T2w_mean_kCALc[2,:,:,:]
        T2w_k_pre04=T2w_mean_kCALc[3,:,:,:]
        T2w_k_pre05=T2w_mean_kCALc[4,:,:,:]
        
        kspace_out1=kspace[:,0,:,:]
        kspace_out2=kspace[:,1,:,:]
        kspace_out3=kspace[:,2,:,:]
        kspace_out4=kspace[:,3,:,:]
        kspace_out5=kspace[:,4,:,:]
        
    

        #loss_kspace0=criterion1(T2w_mean_k_mask_input.real,kspace_mask.real)+criterion1(T2w_mean_k_mask_input.imag,kspace_mask.imag)

        loss_kspace11=criterion1(T2w_k_cal01[mask111>0].real,kspace_out1[mask111>0].real)+criterion1(T2w_k_cal01[mask111>0].imag,kspace_out1[mask111>0].imag)
        loss_kspace12=criterion1(T2w_k_cal02[mask112>0].real,kspace_out2[mask112>0].real)+criterion1(T2w_k_cal02[mask112>0].imag,kspace_out2[mask112>0].imag)
        loss_kspace13=criterion1(T2w_k_cal03[mask113>0].real,kspace_out3[mask113>0].real)+criterion1(T2w_k_cal03[mask113>0].imag,kspace_out3[mask113>0].imag)
        loss_kspace14=criterion1(T2w_k_cal04[mask114>0].real,kspace_out4[mask114>0].real)+criterion1(T2w_k_cal04[mask114>0].imag,kspace_out4[mask114>0].imag)
        loss_kspace15=criterion1(T2w_k_cal05[mask115>0].real,kspace_out5[mask115>0].real)+criterion1(T2w_k_cal05[mask115>0].imag,kspace_out5[mask115>0].imag)
        
        loss_kspace1=criterion1(T2w_k_cal01[mask111>0].real,T2w_k_pre01[mask111>0].real)+criterion1(T2w_k_cal01[mask111>0].imag,T2w_k_pre01[mask111>0].imag)
        loss_kspace2=criterion1(T2w_k_cal02[mask112>0].real,T2w_k_pre02[mask112>0].real)+criterion1(T2w_k_cal02[mask112>0].imag,T2w_k_pre02[mask112>0].imag)
        loss_kspace3=criterion1(T2w_k_cal03[mask113>0].real,T2w_k_pre03[mask113>0].real)+criterion1(T2w_k_cal03[mask113>0].imag,T2w_k_pre03[mask113>0].imag)
        loss_kspace4=criterion1(T2w_k_cal04[mask114>0].real,T2w_k_pre04[mask114>0].real)+criterion1(T2w_k_cal04[mask114>0].imag,T2w_k_pre04[mask114>0].imag)
        loss_kspace5=criterion1(T2w_k_cal05[mask115>0].real,T2w_k_pre05[mask115>0].real)+criterion1(T2w_k_cal05[mask115>0].imag,T2w_k_pre05[mask115>0].imag)
        
        loss = (loss_kspace11+2*loss_kspace12+4*loss_kspace13+6*loss_kspace14+8*loss_kspace15)+(loss_kspace1+loss_kspace2+loss_kspace3+loss_kspace4+loss_kspace5)
        
        
        optimizer.zero_grad()   
        loss.backward()
        optimizer.step()

        train_epoch_loss +=loss.item()
        if (i)%2==0:
            print('epoch:%d - %d, loss:%.10f'%(epoch+1,i,loss.item()))
    for param_group in optimizer.param_groups:
        writer.add_scalar('learning_rate', param_group['lr'], epoch + 1)
    scheduler.step()
    train_epoch_loss /=train_step
    writer.add_scalar('train/loss',train_epoch_loss,epoch+1)
    if (epoch + 1) % 10 == 0:
        plot_2d_or_3d_image(x_t2w1, step=epoch+1, writer=writer, tag="train/input1syn")
        plot_2d_or_3d_image(x_t2w2, step=epoch+1, writer=writer, tag="train/input2syn")
        plot_2d_or_3d_image(x_t2w3, step=epoch+1, writer=writer, tag="train/input3syn")
        plot_2d_or_3d_image(x_t2w4, step=epoch+1, writer=writer, tag="train/input4syn")
        plot_2d_or_3d_image(x_t2w5, step=epoch+1, writer=writer, tag="train/input5syn")
        plot_2d_or_3d_image(T2w[:,0,:,:], step=epoch+1, writer=writer, tag="train/input1")
        plot_2d_or_3d_image(T2w[:,2,:,:], step=epoch+1, writer=writer, tag="train/input3")
        plot_2d_or_3d_image(T2w[:,4,:,:], step=epoch+1, writer=writer, tag="train/input5")
        plot_2d_or_3d_image(T2w_ref[:,0,:,:], step=epoch+1, writer=writer, tag="train/input1ref")
        plot_2d_or_3d_image(T2w_ref[:,2,:,:], step=epoch+1, writer=writer, tag="train/input3ref")
        plot_2d_or_3d_image(T2w_ref[:,4,:,:], step=epoch+1, writer=writer, tag="train/input5ref")
        plot_2d_or_3d_image(image_pre1, step=epoch+1, writer=writer, tag="train/input1predict")
        plot_2d_or_3d_image(image_pre3, step=epoch+1, writer=writer, tag="train/input3predict")
        plot_2d_or_3d_image(image_pre5, step=epoch+1, writer=writer, tag="train/input5predict")
        plot_2d_or_3d_image(target_t2map, step=epoch+1, writer=writer, tag="train/labelt2map")
        plot_2d_or_3d_image(target_pdmap, step=epoch+1, writer=writer, tag="train/labelpdmap")
        plot_2d_or_3d_image(t2m, step=epoch+1, writer=writer, tag="train/outt2map")
        #plot_2d_or_3d_image(M0_show, step=epoch+1, writer=writer, tag="train/outpdmap")


    print(f"epoch {epoch + 1} average loss: {train_epoch_loss:.4f} time elapsed: {(time.time()-tic)/60:.2f} mins")
    torch.cuda.empty_cache()
    if (epoch + 1) % 10 == 0:
        model.eval()     # evaluation
        print('\n testing...')
        time_start=time.time()
        i=0
        val_epoch_loss = 0
        val_step = 0
        val_losses = list()
        #case_ssims_pdm = []
        #case_ssims_t1m = []
        case_ssims_t2m = []
        #case_psnrs_pdm = []
        #case_psnrs_t1m = []
        case_psnrs_t2m = []
        for valdata in val_loader:
            #ssims_t1m = []
            #psnrs_t1m = []
            ssims_t2m = []
            psnrs_t2m = []
            #ssims_pdm = []
            #psnrs_pdm = []
            i+=1
            val_step +=1
            with torch.no_grad():
                target_t2map, target_pdmap, target_t1map = (valdata["t2m"], valdata["pdm"], valdata["t1m"])
                for j in range(target_t2map.shape[-1]):
                    
                    pd=target_pdmap[:,0,:,:,j]
                    t1=target_t1map[:,0,:,:,j]
                    t2=target_t2map[:,0,:,:,j]
                    t2_cal=target_t2map[:,:,:,:,j]
                    pd_cal=target_pdmap[:,:,:,:,j]
                    mask = (target_t2map[:,:,:,:,j] > 0).float()

                    pd = torch.abs(pd)
                    # 将 t2map_slice 和 t1map_slice 中的0替换为一个极小的正数
                    epsilon = 1e-8
                    t2 = t2.clamp(min=epsilon)
                    t1 = t1.clamp(min=epsilon)

                    ETL=5
                    mean=0
                    sigma=0.05
                    T2w_mean_i=torch.rand(ETL,t2.shape[-3],t2.shape[-2],t2.shape[-1])
                    T2w_mean_i=T2w_mean_i.to('cuda')
                    real=torch.rand(ETL,t2.shape[-3],t2.shape[-2],t2.shape[-1])
                    imag=torch.rand(ETL,t2.shape[-3],t2.shape[-2],t2.shape[-1])
                    T2w_mean_k = torch.complex(real,imag)
                    T2w_mean_k=T2w_mean_k.to('cuda')
                    T2w_mean_k1 = torch.complex(real,imag)
                    T2w_mean_k1=T2w_mean_k1.to('cuda')
                    real1=torch.rand(1,t2.shape[-3],t2.shape[-2],t2.shape[-1])
                    imag1=torch.rand(1,t2.shape[-3],t2.shape[-2],t2.shape[-1])
                    T2w_k = torch.complex(real1,imag1)
                    T2w_k=T2w_k.to('cuda')
                    TR=6000
                    esp=50

                    #mask1=torch.rand(ETL,256,256)
                    for m in range(ETL):
                        T2w_mean_i[m,:,:,:] = torch.abs(pd * torch.exp(-esp*(m+1) / t2) * (1 -2*torch.exp(-(TR-esp*(m+1)/2) / t1)+ torch.exp(-TR / t1)))
                    #T2w_mean_imax=torch.max(T2w_mean_i)
                    
                    T2w_mean_imax=torch.amax(T2w_mean_i,dim=(0,2,3))
                    T2w_mean_imax=T2w_mean_imax+epsilon
                    T2w_mean_imax_expand=T2w_mean_imax.unsqueeze(0).unsqueeze(2).unsqueeze(3)
                    T2w_mean_i=T2w_mean_i/T2w_mean_imax_expand
                    T2w_mean_k1=torch.fft.fftshift(torch.fft.fft2((torch.fft.fftshift(T2w_mean_i, dim=(-2,-1))), dim=(-2,-1)), dim=(-2,-1))
                    for m in range(T2w_mean_k1.shape[1]):
                        temp=T2w_mean_k1[:,m,:,:]
                        data_std=torch.std(temp)
                        T2w_mean_k[:,m,:,:] =T2w_mean_k1[:,m,:,:]+(torch.complex(torch.normal(mean, sigma*data_std, temp.shape),torch.normal(mean, sigma*data_std, temp.shape))).to('cuda')
                    
                    mask11=mask1.unsqueeze(1)
                    mask11=mask11.repeat(1,T2w_mean_k.shape[1], 1, 1) 
                    mask11 = mask11.to(device)
                    mask11 = mask11.to(torch.float32)
                    #print(mask11[0,0,:,1])

                    T2w_mean_k_mask=T2w_mean_k*mask11
                    #print(T2w_mean_k_mask[0,0,:,1])
                    
                    T2w=torch.abs(torch.fft.fftshift(torch.fft.ifft2((torch.fft.fftshift(T2w_mean_k_mask, dim=(-2,-1))), dim=(-2,-1)), dim=(-2,-1)))    
                    T2w=T2w.permute(1,0,2,3)
                    T2w_ref=torch.abs(torch.fft.fftshift(torch.fft.ifft2((torch.fft.fftshift(T2w_mean_k, dim=(-2,-1))), dim=(-2,-1)), dim=(-2,-1))) 
                    T2w_ref=T2w_ref.permute(1,0,2,3)
                    T2w_mean_k_mask_input=T2w_mean_k_mask.permute(1,0,2,3)
                    mask_input=mask11.permute(1,0,2,3)

                    t2m, image, kspace = model(T2w_mean_k_mask_input.to(device),mask_input.to(device))
                    t2m = torch.relu(t2m)
                    image = torch.relu(image)
                    
                    kspace_mask=kspace*mask_input



                    T2w_mean_iCAL=torch.rand(ETL,t2.shape[-3],1,t2.shape[-2],t2.shape[-1])
                    T2w_mean_iCAL=T2w_mean_iCAL.to('cuda')
                    realCAL=torch.rand(ETL,t2.shape[-3],1,t2.shape[-2],t2.shape[-1])
                    imagCAL=torch.rand(ETL,t2.shape[-3],1,t2.shape[-2],t2.shape[-1])
                    T2w_mean_kCAL = torch.complex(realCAL,imagCAL)
                    T2w_mean_kCAL=T2w_mean_kCAL.to('cuda')

                    T2w_mean_iCAL[0,:,:,:,:] = image[:,:,0,:,:]
                    T2w_mean_kCAL[0,:,:,:,:] =torch.fft.fftshift(torch.fft.fft2((torch.fft.fftshift(T2w_mean_iCAL[0,:,:,:,:], dim=(-2,-1))), dim=(-2,-1)), dim=(-2,-1))
                    
                    M0_updata=image[:,:,0,:,:]
                    M0_show=M0_updata.clone()
                    M0_show[mask == 0] = 0  
                        
                    for m in range(ETL-1):
                        T2w_mean_iCAL[m+1,:,:,:,:] = torch.abs(M0_updata * torch.exp(-esp*(m+1) / t2m))
                        T2w_mean_kCAL[m+1,:,:,:,:] =torch.fft.fftshift(torch.fft.fft2((torch.fft.fftshift(T2w_mean_iCAL[m+1,:,:,:,:], dim=(-2,-1))), dim=(-2,-1)), dim=(-2,-1))
                    T2w_mean_kCALc=T2w_mean_kCAL[:,:,0,:,:]
                    
                    image_pre1=T2w_mean_iCAL[0,:,:,:,:]
                    image_pre2=T2w_mean_iCAL[1,:,:,:,:]
                    image_pre3=T2w_mean_iCAL[2,:,:,:,:]
                    image_pre4=T2w_mean_iCAL[3,:,:,:,:]
                    image_pre5=T2w_mean_iCAL[4,:,:,:,:]
                    image_pre1[mask==0]=0
                    image_pre2[mask==0]=0
                    image_pre3[mask==0]=0
                    image_pre4[mask==0]=0
                    image_pre5[mask==0]=0
                    
                    x_t2w1=image[:,:,0,:,:].clone()
                    x_t2w2=image[:,:,1,:,:].clone()
                    x_t2w3=image[:,:,2,:,:].clone()
                    x_t2w4=image[:,:,3,:,:].clone()
                    x_t2w5=image[:,:,4,:,:].clone()
                    x_t2w1[mask==0]=0
                    x_t2w2[mask==0]=0
                    x_t2w3[mask==0]=0
                    x_t2w4[mask==0]=0
                    x_t2w5[mask==0]=0
                    
                    mask111=mask11[0,:,:,:]
                    mask112=mask11[1,:,:,:]
                    mask113=mask11[2,:,:,:]
                    mask114=mask11[3,:,:,:]
                    mask115=mask11[4,:,:,:]
                    T2w_k_cal01=T2w_mean_k[0,:,:,:]
                    T2w_k_cal02=T2w_mean_k[1,:,:,:]
                    T2w_k_cal03=T2w_mean_k[2,:,:,:]
                    T2w_k_cal04=T2w_mean_k[3,:,:,:]
                    T2w_k_cal05=T2w_mean_k[4,:,:,:]
                    T2w_k_pre01=T2w_mean_kCALc[0,:,:,:]
                    T2w_k_pre02=T2w_mean_kCALc[1,:,:,:]
                    T2w_k_pre03=T2w_mean_kCALc[2,:,:,:]
                    T2w_k_pre04=T2w_mean_kCALc[3,:,:,:]
                    T2w_k_pre05=T2w_mean_kCALc[4,:,:,:]
                    
                    kspace_out1=kspace[:,0,:,:]
                    kspace_out2=kspace[:,1,:,:]
                    kspace_out3=kspace[:,2,:,:]
                    kspace_out4=kspace[:,3,:,:]
                    kspace_out5=kspace[:,4,:,:]
                    


                    #loss_kspace0=criterion1(T2w_mean_k_mask_input.real,kspace_mask.real)+criterion1(T2w_mean_k_mask_input.imag,kspace_mask.imag)

                    loss_kspace11=criterion1(T2w_k_cal01[mask111>0].real,kspace_out1[mask111>0].real)+criterion1(T2w_k_cal01[mask111>0].imag,kspace_out1[mask111>0].imag)
                    loss_kspace12=criterion1(T2w_k_cal02[mask112>0].real,kspace_out2[mask112>0].real)+criterion1(T2w_k_cal02[mask112>0].imag,kspace_out2[mask112>0].imag)
                    loss_kspace13=criterion1(T2w_k_cal03[mask113>0].real,kspace_out3[mask113>0].real)+criterion1(T2w_k_cal03[mask113>0].imag,kspace_out3[mask113>0].imag)
                    loss_kspace14=criterion1(T2w_k_cal04[mask114>0].real,kspace_out4[mask114>0].real)+criterion1(T2w_k_cal04[mask114>0].imag,kspace_out4[mask114>0].imag)
                    loss_kspace15=criterion1(T2w_k_cal05[mask115>0].real,kspace_out5[mask115>0].real)+criterion1(T2w_k_cal05[mask115>0].imag,kspace_out5[mask115>0].imag)
                    
                    loss_kspace1=criterion1(T2w_k_cal01[mask111>0].real,T2w_k_pre01[mask111>0].real)+criterion1(T2w_k_cal01[mask111>0].imag,T2w_k_pre01[mask111>0].imag)
                    loss_kspace2=criterion1(T2w_k_cal02[mask112>0].real,T2w_k_pre02[mask112>0].real)+criterion1(T2w_k_cal02[mask112>0].imag,T2w_k_pre02[mask112>0].imag)
                    loss_kspace3=criterion1(T2w_k_cal03[mask113>0].real,T2w_k_pre03[mask113>0].real)+criterion1(T2w_k_cal03[mask113>0].imag,T2w_k_pre03[mask113>0].imag)
                    loss_kspace4=criterion1(T2w_k_cal04[mask114>0].real,T2w_k_pre04[mask114>0].real)+criterion1(T2w_k_cal04[mask114>0].imag,T2w_k_pre04[mask114>0].imag)
                    loss_kspace5=criterion1(T2w_k_cal05[mask115>0].real,T2w_k_pre05[mask115>0].real)+criterion1(T2w_k_cal05[mask115>0].imag,T2w_k_pre05[mask115>0].imag)
                    
                    loss = (loss_kspace11+2*loss_kspace12+4*loss_kspace13+6*loss_kspace14+8*loss_kspace15)+(loss_kspace1+loss_kspace2+loss_kspace3+loss_kspace4+loss_kspace5)
                    
                    val_losses.append(loss.item())
                    
                    t2_cal[mask == 0] = 0
                    t2m[mask == 0] = 0
                    tar_t2m_np = t2_cal.cpu().numpy()
                    tar_t2m_np=tar_t2m_np[:,0,...]
                    t2m_np = t2m.data.cpu().numpy()
                    t2m_np=t2m_np[:,0,...]
                    min_val_t2m = min(tar_t2m_np.min(), t2m_np.min())
                    max_val_t2m = max(tar_t2m_np.max(), t2m_np.max())
                    tar_t2m_np_normalized = normalize_to_range(tar_t2m_np, min_val_t2m, max_val_t2m)
                    t2m_np_normalized = normalize_to_range(t2m_np, min_val_t2m, max_val_t2m)
                    t2m_ssim = skimage_ssim(tar_t2m_np_normalized, t2m_np_normalized)
                    ssims_t2m.append(t2m_ssim)
                    t2m_psnr=psnr(tar_t2m_np_normalized, t2m_np_normalized)
                    psnrs_t2m.append(t2m_psnr)
                    

                case_ssims_t2m.append(torch.mean(torch.tensor(ssims_t2m)))
                case_psnrs_t2m.append(torch.mean(torch.tensor(psnrs_t2m)))
                
                if (i)%2==0:
                    print('epoch:%d - %d, loss:%.10f'%(epoch+1,i,loss.item()))
        ssim_t2m = torch.mean(torch.tensor(case_ssims_t2m)) 
        psnr_t2m = torch.mean(torch.tensor(case_psnrs_t2m))
        valid_loss = torch.mean(torch.tensor(val_losses))
        writer.add_scalar('valid/ssim_t2m',ssim_t2m,epoch+1)
        writer.add_scalar('valid/psnr_t2m',psnr_t2m,epoch+1)
        writer.add_scalar('valid/loss',valid_loss,epoch+1)

        
        if ssim_t2m >= best_ssim_all:
            best_ssim_all = ssim_t2m
            torch.save(model, os.path.join(output_dir, 'BestSsim_epoch-%d-%.10f.pth' % (epoch+1, best_ssim_all)))
        if psnr_t2m >= best_psnr_all:
            best_psnr_all = psnr_t2m
            torch.save(model, os.path.join(output_dir, 'BestPsnr_epoch-%d-%.10f.pth' % (epoch+1, best_psnr_all)))  
        time_end=time.time()
        print('time cost for testing',time_end-time_start,'s')
        #torch.save(model, os.path.join(output_dir, 'epoch-%d-%.10f.pth' % (epoch+1, val_epoch_loss)))   

writer.close()
print('Finished Training')