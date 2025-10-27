import torch
import torch.nn as nn
def add_conv_stage(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=True, useBN=False):
  if useBN:
    return nn.Sequential(
      nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
      nn.BatchNorm2d(dim_out),
      nn.LeakyReLU(0.1),
      nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
      nn.BatchNorm2d(dim_out),
      nn.LeakyReLU(0.1)
    )
  else:
    return nn.Sequential(
      nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
      nn.ReLU(),
      nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
      nn.ReLU()
    )
       
def upsample(ch_coarse, ch_fine):
  return nn.Sequential(
    nn.ConvTranspose2d(ch_coarse, ch_fine, 4, 2, 1, bias=False),
    nn.ReLU()
  )
  
class QSE(nn.Module):
  def __init__(self, useBN=True):
    super(QSE, self).__init__()
    
    #I1learning
    self.conv1_t2w0 = add_conv_stage(5, 32, useBN=useBN)
    self.conv2_t2w0 = add_conv_stage(32, 64, useBN=useBN)
    self.conv3_t2w0 = add_conv_stage(64, 128, useBN=useBN)
    self.conv4_t2w0 = add_conv_stage(128, 256, useBN=useBN)
    
    self.conv0 = add_conv_stage(256, 256, kernel_size=1, stride=1, padding=0, bias=True, useBN=useBN)
    self.conv4m0 = add_conv_stage(128+128, 128, useBN=useBN)
    self.conv3m0 = add_conv_stage(64+64, 64, useBN=useBN)
    self.conv2m0 = add_conv_stage(32+32, 32, useBN=useBN)
    self.upsample540 = upsample(256, 128)
    self.upsample430 = upsample(128, 64)
    self.upsample320 = upsample(64, 32)
    self.conv0_pdmap = nn.Conv2d(32, 1, 3, 1, 1)
    
    self.conv1 = add_conv_stage(256, 256, kernel_size=1, stride=1, padding=0, bias=True, useBN=useBN)
    self.conv4m1 = add_conv_stage(128+128, 128, useBN=useBN)
    self.conv3m1 = add_conv_stage(64+64, 64, useBN=useBN)
    self.conv2m1 = add_conv_stage(32+32, 32, useBN=useBN)
    self.upsample541 = upsample(256, 128)
    self.upsample431 = upsample(128, 64)
    self.upsample321 = upsample(64, 32)    
    self.conv0_pd1map = nn.Conv2d(32, 1, 3, 1, 1)

    self.conv2 = add_conv_stage(256, 256, kernel_size=1, stride=1, padding=0, bias=True, useBN=useBN)
    self.conv4m2 = add_conv_stage(128+128, 128, useBN=useBN)
    self.conv3m2 = add_conv_stage(64+64, 64, useBN=useBN)
    self.conv2m2 = add_conv_stage(32+32, 32, useBN=useBN)
    self.upsample542 = upsample(256, 128)
    self.upsample432 = upsample(128, 64)
    self.upsample322 = upsample(64, 32)     
    self.conv0_pd2map = nn.Conv2d(32, 1, 3, 1, 1)

    self.conv3 = add_conv_stage(256, 256, kernel_size=1, stride=1, padding=0, bias=True, useBN=useBN)
    self.conv4m3 = add_conv_stage(128+128, 128, useBN=useBN)
    self.conv3m3 = add_conv_stage(64+64, 64, useBN=useBN)
    self.conv2m3 = add_conv_stage(32+32, 32, useBN=useBN)
    self.upsample543 = upsample(256, 128)
    self.upsample433 = upsample(128, 64)
    self.upsample323 = upsample(64, 32)     
    self.conv0_pd3map = nn.Conv2d(32, 1, 3, 1, 1)

    self.conv4 = add_conv_stage(256, 256, kernel_size=1, stride=1, padding=0, bias=True, useBN=useBN)
    self.conv4m4 = add_conv_stage(128+128, 128, useBN=useBN)
    self.conv3m4 = add_conv_stage(64+64, 64, useBN=useBN)
    self.conv2m4 = add_conv_stage(32+32, 32, useBN=useBN)
    self.upsample544 = upsample(256, 128)
    self.upsample434 = upsample(128, 64)
    self.upsample324 = upsample(64, 32)     
    self.conv0_pd4map = nn.Conv2d(32, 1, 3, 1, 1)
    # T2W1 branch
    self.conv1_t2w1 = add_conv_stage(1, 32, useBN=useBN)
    self.conv2_t2w1 = add_conv_stage(32, 64, useBN=useBN)
    self.conv3_t2w1 = add_conv_stage(64, 128, useBN=useBN)
    self.conv4_t2w1 = add_conv_stage(128, 256, useBN=useBN)

    # T2W2 branch
    self.conv1_t2w2 = add_conv_stage(1, 32, useBN=useBN)
    self.conv2_t2w2 = add_conv_stage(32, 64, useBN=useBN)
    self.conv3_t2w2 = add_conv_stage(64, 128, useBN=useBN)
    self.conv4_t2w2 = add_conv_stage(128, 256, useBN=useBN)

    # T2w3 branch
    self.conv1_t2w3 = add_conv_stage(1, 32, useBN=useBN)
    self.conv2_t2w3 = add_conv_stage(32, 64, useBN=useBN)
    self.conv3_t2w3 = add_conv_stage(64, 128, useBN=useBN)
    self.conv4_t2w3 = add_conv_stage(128, 256, useBN=useBN)

    # T2W4 branch
    self.conv1_t2w4 = add_conv_stage(1, 32, useBN=useBN)
    self.conv2_t2w4 = add_conv_stage(32, 64, useBN=useBN)
    self.conv3_t2w4 = add_conv_stage(64, 128, useBN=useBN)
    self.conv4_t2w4 = add_conv_stage(128, 256, useBN=useBN)

    # T2w5 branch
    self.conv1_t2w5 = add_conv_stage(1, 32, useBN=useBN)
    self.conv2_t2w5 = add_conv_stage(32, 64, useBN=useBN)
    self.conv3_t2w5 = add_conv_stage(64, 128, useBN=useBN)
    self.conv4_t2w5 = add_conv_stage(128, 256, useBN=useBN)


    self.conv = add_conv_stage(256*5, 256*5, kernel_size=1, stride=1, padding=0, bias=True, useBN=useBN)

    self.conv4m = add_conv_stage(256+128*5, 256, useBN=useBN)
    self.conv3m = add_conv_stage(128+64*5, 128, useBN=useBN)
    self.conv2m = add_conv_stage(64+32*5, 64, useBN=useBN)

     # 单输出改为多输出
    self.conv0_t2map = nn.Conv2d(64, 1, 3, 1, 1)
    

    self.max_pool = nn.MaxPool2d(2)
    self.upsample54 = upsample(256*5, 256)
    self.upsample43 = upsample(256, 128)
    self.upsample32 = upsample(128, 64)

    # Weight initialization
    for m in self.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            if m.bias is not None:
                m.bias.data.zero_()

  def forward(self, k_t2w, mask):

      x_t2w0=torch.abs(torch.fft.fftshift(torch.fft.ifft2((torch.fft.fftshift(k_t2w, dim=(-2,-1))), dim=(-2,-1)), dim=(-2,-1)))

      #T2W Image learning
      conv1_out_t2w0 = self.conv1_t2w0(x_t2w0)
      conv2_out_t2w0 = self.conv2_t2w0(self.max_pool(conv1_out_t2w0))
      conv3_out_t2w0 = self.conv3_t2w0(self.max_pool(conv2_out_t2w0))
      conv4_out_t2w0 = self.conv4_t2w0(self.max_pool(conv3_out_t2w0))
      
      combined_features0 = self.conv0(conv4_out_t2w0)
      conv5m_out0 = torch.cat((self.upsample540(combined_features0), conv3_out_t2w0), 1) #256+128*3=
      conv4m_out0 = self.conv4m0(conv5m_out0)  #256
      conv4m_out_0 = torch.cat((self.upsample430(conv4m_out0), conv2_out_t2w0), 1)#256+64*3
      conv3m_out0 = self.conv3m0(conv4m_out_0)#128
      conv3m_out_0 = torch.cat((self.upsample320(conv3m_out0), conv1_out_t2w0), 1)#64*4
      conv2m_out0 = self.conv2m0(conv3m_out_0)#64
      conv0_pdmap_out = self.conv0_pdmap(conv2m_out0)
      
      combined_features1 = self.conv1(conv4_out_t2w0)
      conv5m_out1 = torch.cat((self.upsample541(combined_features1), conv3_out_t2w0), 1) #256+128*3=
      conv4m_out1 = self.conv4m1(conv5m_out1)  #256
      conv4m_out_1 = torch.cat((self.upsample431(conv4m_out1), conv2_out_t2w0), 1)#256+64*3
      conv3m_out1 = self.conv3m1(conv4m_out_1)#128
      conv3m_out_1 = torch.cat((self.upsample321(conv3m_out1), conv1_out_t2w0), 1)#64*4
      conv2m_out1 = self.conv2m1(conv3m_out_1)#64      
      conv0_pd1map_out = self.conv0_pd1map(conv2m_out1)

      combined_features2 = self.conv2(conv4_out_t2w0)
      conv5m_out2 = torch.cat((self.upsample542(combined_features2), conv3_out_t2w0), 1) #256+128*3=
      conv4m_out2 = self.conv4m2(conv5m_out2)  #256
      conv4m_out_2 = torch.cat((self.upsample432(conv4m_out2), conv2_out_t2w0), 1)#256+64*3
      conv3m_out2 = self.conv3m2(conv4m_out_2)#128
      conv3m_out_2 = torch.cat((self.upsample322(conv3m_out2), conv1_out_t2w0), 1)#64*4
      conv2m_out2 = self.conv2m2(conv3m_out_2)#64        
      conv0_pd2map_out = self.conv0_pd2map(conv2m_out2)
      
      combined_features3 = self.conv3(conv4_out_t2w0)
      conv5m_out3 = torch.cat((self.upsample543(combined_features3), conv3_out_t2w0), 1) #256+128*3=
      conv4m_out3 = self.conv4m3(conv5m_out3)  #256
      conv4m_out_3 = torch.cat((self.upsample433(conv4m_out3), conv2_out_t2w0), 1)#256+64*3
      conv3m_out3 = self.conv3m3(conv4m_out_3)#128
      conv3m_out_3 = torch.cat((self.upsample323(conv3m_out3), conv1_out_t2w0), 1)#64*4
      conv2m_out3 = self.conv2m3(conv3m_out_3)#64         
      conv0_pd3map_out = self.conv0_pd3map(conv2m_out3)
      
      combined_features4 = self.conv4(conv4_out_t2w0)
      conv5m_out4 = torch.cat((self.upsample544(combined_features4), conv3_out_t2w0), 1) #256+128*3=
      conv4m_out4 = self.conv4m4(conv5m_out4)  #256
      conv4m_out_4 = torch.cat((self.upsample434(conv4m_out4), conv2_out_t2w0), 1)#256+64*3
      conv3m_out4 = self.conv3m4(conv4m_out_4)#128
      conv3m_out_4 = torch.cat((self.upsample324(conv3m_out4), conv1_out_t2w0), 1)#64*4
      conv2m_out4 = self.conv2m4(conv3m_out_4)#64        
      conv0_pd4map_out = self.conv0_pd4map(conv2m_out4)
      
      x_t2w_out = torch.cat((conv0_pdmap_out, conv0_pd1map_out, conv0_pd2map_out, conv0_pd3map_out, conv0_pd4map_out), 1)#64*4
      x_t2w_out_res=x_t2w_out+x_t2w0
      k_t2w_out=torch.fft.fftshift(torch.fft.fft2((torch.fft.fftshift(x_t2w_out_res, dim=(-2,-1))), dim=(-2,-1)), dim=(-2,-1))
      k_t2w_out1=k_t2w_out*(1-mask)+k_t2w
      x_t2w_out1=torch.abs(torch.fft.fftshift(torch.fft.ifft2((torch.fft.fftshift(k_t2w_out1, dim=(-2,-1))), dim=(-2,-1)), dim=(-2,-1)))
      
      x_t2w=x_t2w_out1.unsqueeze(1)
      x_t2w1=x_t2w[:,:,0,:,:]
      x_t2w2=x_t2w[:,:,1,:,:]
      x_t2w3=x_t2w[:,:,2,:,:]
      x_t2w4=x_t2w[:,:,3,:,:]
      x_t2w5=x_t2w[:,:,4,:,:]
      
      
      # T2W1 branch
      conv1_out_t2w1 = self.conv1_t2w1(x_t2w1)
      conv2_out_t2w1 = self.conv2_t2w1(self.max_pool(conv1_out_t2w1))
      conv3_out_t2w1 = self.conv3_t2w1(self.max_pool(conv2_out_t2w1))
      conv4_out_t2w1 = self.conv4_t2w1(self.max_pool(conv3_out_t2w1))
      # T2W2 branch
      conv1_out_t2w2 = self.conv1_t2w2(x_t2w2)
      conv2_out_t2w2 = self.conv2_t2w2(self.max_pool(conv1_out_t2w2))
      conv3_out_t2w2 = self.conv3_t2w2(self.max_pool(conv2_out_t2w2))
      conv4_out_t2w2 = self.conv4_t2w2(self.max_pool(conv3_out_t2w2))
      # T2W3 branch
      conv1_out_t2w3 = self.conv1_t2w3(x_t2w3)
      conv2_out_t2w3 = self.conv2_t2w3(self.max_pool(conv1_out_t2w3))
      conv3_out_t2w3 = self.conv3_t2w3(self.max_pool(conv2_out_t2w3))
      conv4_out_t2w3 = self.conv4_t2w3(self.max_pool(conv3_out_t2w3))
      # T2W4 branch
      conv1_out_t2w4 = self.conv1_t2w4(x_t2w4)
      conv2_out_t2w4 = self.conv2_t2w4(self.max_pool(conv1_out_t2w4))
      conv3_out_t2w4 = self.conv3_t2w4(self.max_pool(conv2_out_t2w4))
      conv4_out_t2w4 = self.conv4_t2w4(self.max_pool(conv3_out_t2w4))
      # T2W5 branch
      conv1_out_t2w5 = self.conv1_t2w5(x_t2w5)
      conv2_out_t2w5 = self.conv2_t2w5(self.max_pool(conv1_out_t2w5))
      conv3_out_t2w5 = self.conv3_t2w5(self.max_pool(conv2_out_t2w5))
      conv4_out_t2w5 = self.conv4_t2w5(self.max_pool(conv3_out_t2w5))
      # Concatenate features from all branches
      combined_features = torch.cat((conv4_out_t2w1, conv4_out_t2w2, conv4_out_t2w3, conv4_out_t2w4, conv4_out_t2w5), dim=1) #256*3
      combined_features1 = self.conv(combined_features)
      conv5m_out = torch.cat((self.upsample54(combined_features1), conv3_out_t2w1, conv3_out_t2w2, conv3_out_t2w3, conv3_out_t2w4, conv3_out_t2w5), 1) #256+128*3=
      conv4m_out = self.conv4m(conv5m_out)  #256

      conv4m_out_ = torch.cat((self.upsample43(conv4m_out), conv2_out_t2w1, conv2_out_t2w2, conv2_out_t2w3, conv2_out_t2w4, conv2_out_t2w5), 1)#256+64*3
      conv3m_out = self.conv3m(conv4m_out_)#128

      conv3m_out_ = torch.cat((self.upsample32(conv3m_out), conv1_out_t2w1, conv1_out_t2w2, conv1_out_t2w3, conv1_out_t2w4, conv1_out_t2w5), 1)#64*4
      conv2m_out = self.conv2m(conv3m_out_)#64

      conv0_t2map_out = self.conv0_t2map(conv2m_out)
      

      return conv0_t2map_out, x_t2w, k_t2w_out
