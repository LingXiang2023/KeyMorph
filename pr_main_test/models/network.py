import torch
import torch.nn as nn
import random
import config as c
from utils.image import quantization
#just take a try
import torch.nn.functional as F
def print_activate_params(model):
    ans=0
    for name, module in model.named_children():
        if isinstance(module, ConvExpert):
         
            ans+=module.hidingactivate.numel()*module.hidingactivate.element_size() 
            ans+=module.recoveractivate.numel()*module.recoveractivate.element_size() 
            ans+=module.denoise.numel()*module.denoise.element_size() 
            if module.strange:
             ans+=module.coveractivate.numel()*module.coveractivate.element_size() 
        else: ans+=print_activate_params(module)
    return ans
class LinearExpert(nn.Module):
    def __init__(self, in_features, out_features, p=4):
        super(LinearExpert, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.hidingactivate = nn.Parameter(torch.tensor([1, 0.9, 0.1, 0], dtype=torch.float32))
        self.recoveractivate = nn.Parameter(torch.tensor([0, 0.1, 0.9, 1], dtype=torch.float32))
        self.p = p
        self.split = self._create_mask(p)

    def _create_mask(self, p):
        ran = torch.rand_like(self.linear.weight)
        split = []
        for i in range(p):
            mask = (i / p <= ran) & (ran < (i + 1) / p)
            split.append(mask.float())
        return split

    def forward(self, x, mode='denoise'):
        if mode == 'denoise':
            return self.linear(x)
        elif mode == 'hiding':
            ratio = self.hidingactivate
            ans = 0
            device=self.linear.weight.device
            for i in range(self.p):
                mask=self.split[i].to(device)
                sparse_weights = self.linear.weight * mask
                ans += F.linear(x, sparse_weights, self.linear.bias) * ratio[i]
            return ans
        else:
            ratio = self.recoveractivate
            ans = 0
            device=self.linear.weight.device
            for i in range(self.p):
                mask=self.split[i].to(device)
                sparse_weights = self.linear.weight * mask
                ans += F.linear(x, sparse_weights, self.linear.bias) * ratio[i]
            return ans
class afterConvExpert(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, p=4,strange=False):
        super(afterConvExpert, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,bias=False)
        self.denoise = nn.Parameter(torch.tensor([0.1, 0.1, 0.1, 1], dtype=torch.float32))
        self.p = p
        self.split = self._create_mask(p)
        self.clear=True
        self.bias1 = nn.Parameter(torch.zeros(out_channels))
        self.bias2 = nn.Parameter(torch.zeros(out_channels))
        self.bias3 = nn.Parameter(torch.zeros(out_channels))
    def _create_mask(self, p):
        ran = torch.rand_like(self.conv.weight)
        split = []
        for i in range(p):
            mask = (i / p <= ran) & (ran < (i + 1) / p)
            split.append(mask.float())
        return split
    '''    
    def forward(self, x, mode='denoise'):
            ratio = F.softmax(self.denoise,dim=0)
            ans = 0
            device=self.conv.weight.device
            for i in range(self.p):
                mask=self.split[i].to(device)
                sparse_weights = self.conv.weight * mask
                ans+=F.conv2d(x, sparse_weights,None, self.conv.stride, self.conv.padding) *ratio[i]
            return ans +self.bias1.view(1, -1, 1, 1) 

    '''
    def forward(self, x, mode='denoise'):
            ratio = F.softmax(self.denoise,dim=0)
            ans = 0
            device=self.conv.weight.device
            now=[]
            with torch.no_grad():
             for i in range(self.p-1):
                mask=self.split[i].to(device)
                sparse_weights = self.conv.weight * mask
                now.append(F.conv2d(x, sparse_weights,None, self.conv.stride, self.conv.padding) )
            ans+=now[0]*ratio[0]+now[1]*ratio[1]+now[2]*ratio[2]
            mask=self.split[3].to(device)
            sparse_weights = self.conv.weight * mask
            ans+=F.conv2d(x, sparse_weights,None, self.conv.stride, self.conv.padding) *ratio[3]
            return ans +self.bias1.view(1, -1, 1, 1) 
class ConvExpert(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, p=4,strange=False):
        super(ConvExpert, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,bias=False)
        if strange:
           self.coveractivate=nn.Parameter(torch.tensor([0.1, 1, 0.1, 0.1], dtype=torch.float32))
        if strange:
            self.hidingactivate = nn.Parameter(torch.tensor([1, 0.1, 0.1, 0.1], dtype=torch.float32))
        else:
            self.hidingactivate = nn.Parameter(torch.tensor([1, 1, 0.1, 0.1], dtype=torch.float32))
        self.recoveractivate = nn.Parameter(torch.tensor([0.1, 0.1, 0.1, 1], dtype=torch.float32))
        self.denoise = nn.Parameter(torch.tensor([0.1, 0.1, 1, 0.1], dtype=torch.float32))
        self.p = p
        self.split = self._create_mask(p)
        self.strange=strange
        self.bias1 = nn.Parameter(torch.zeros(out_channels))
        self.bias2 = nn.Parameter(torch.zeros(out_channels))
        self.bias3 = nn.Parameter(torch.zeros(out_channels))
    def _create_mask(self, p):
        ran = torch.rand_like(self.conv.weight)
        split = []
        for i in range(p):
            mask = (i / p <= ran) & (ran < (i + 1) / p)
            split.append(mask.float())
        return split

    def forward(self, x, mode='denoise',assign=False):
        if mode == 'denoise':
            ratio = F.softmax(self.denoise,dim=0)
            ans = 0
            device=self.conv.weight.device
            for i in range(self.p):
                mask=self.split[i].to(device)
                sparse_weights = self.conv.weight * mask
                ans += F.conv2d(x, sparse_weights,None, self.conv.stride, self.conv.padding) * ratio[i]
            return ans+self.bias1.view(1, -1, 1, 1) 
        elif mode == 'hiding':
            if (assign and self.strange):
               ratio = F.softmax(self.coveractivate,dim=0)
            else:
               ratio = F.softmax(self.hidingactivate,dim=0)
            ans = 0
            device=self.conv.weight.device
            for i in range(self.p):
                mask=self.split[i].to(device)
                sparse_weights = self.conv.weight * mask
                ans += F.conv2d(x, sparse_weights, None,self.conv.stride, self.conv.padding) * ratio[i]
            return ans+self.bias1.view(1, -1, 1, 1) 
        else:
            ratio = F.softmax(self.recoveractivate,dim=0)
            ans = 0
            device=self.conv.weight.device
            for i in range(self.p):
                mask=self.split[i].to(device)
                sparse_weights = self.conv.weight * mask
                ans += F.conv2d(x, sparse_weights,None, self.conv.stride, self.conv.padding) * ratio[i]
            return ans+self.bias1.view(1, -1, 1, 1) 

class preModel(nn.Module):
    def __init__(self):
        super(preModel, self).__init__()
        self.layers = nn.Sequential(
            nn.GroupNorm(16, 256),
            nn.LeakyReLU(),
            ConvExpert(in_channels=256, out_channels=128, kernel_size=3, stride=1 ,padding=1,strange=True), 
            nn.GroupNorm(8, 128),
            nn.LeakyReLU(),
            ConvExpert(in_channels=128, out_channels=256, kernel_size=3, stride=1 ,padding=1,strange=True), 
        )    

    def forward(self, x, mode='denoise',assign=False):
        for layer in self.layers:
            if isinstance(layer, (ConvExpert, LinearExpert)):
              if(assign):
                x=layer(x,mode,assign=True)
              else:
                x = layer(x, mode)
            else:
                x = layer(x)
        return x
class inModel(nn.Module):
    def __init__(self):
        super(inModel, self).__init__()
        self.layers = nn.Sequential(
            ConvExpert(in_channels=3, out_channels=64, kernel_size=3, stride=1 ,padding=1,strange=True), 
            nn.GroupNorm(4, 64),
            nn.LeakyReLU(),
            ConvExpert(in_channels=64, out_channels=128, kernel_size=3, stride=1 ,padding=1), 
            nn.GroupNorm(8, 128),
            nn.LeakyReLU(),
            ConvExpert(in_channels=128, out_channels=256, kernel_size=3, stride=1 ,padding=1,strange=True), 
        )   
    def forward(self, x, mode='denoise',assign=False):
        for layer in self.layers:
            if isinstance(layer, (ConvExpert, LinearExpert)):
              if(assign):
                x=layer(x,mode,assign=True)
              else:
                x = layer(x, mode)
            else:
                x = layer(x)
        return x
class afterModel(nn.Module):
    def __init__(self):
        super(afterModel, self).__init__()
        self.layers = nn.Sequential(
            nn.GroupNorm(16, 256),
            nn.LeakyReLU(),
            ConvExpert(in_channels=256, out_channels=128, kernel_size=3, stride=1 ,padding=1), 
            nn.GroupNorm(8, 128),
            nn.LeakyReLU(),
            ConvExpert(in_channels=128, out_channels=256, kernel_size=3, stride=1 ,padding=1), 
        )    

    def forward(self, x, mode='denoise'):
        for layer in self.layers:
            if isinstance(layer, (ConvExpert, LinearExpert)):
                x = layer(x, mode)
            else:
                x = layer(x)
        return x
class outModel(nn.Module):
    def __init__(self):
        super(outModel, self).__init__()
        self.layers = nn.Sequential(
            nn.GroupNorm(16, 256),
            nn.LeakyReLU(),
            afterConvExpert(in_channels=256, out_channels=128, kernel_size=3, stride=1 ,padding=1), 
            nn.GroupNorm(8, 128),
            nn.LeakyReLU(),
            afterConvExpert(in_channels=128, out_channels=64, kernel_size=3, stride=1 ,padding=1), 
            nn.GroupNorm(4, 64),
            nn.LeakyReLU(),
            afterConvExpert(in_channels=64, out_channels=3, kernel_size=3, stride=1 ,padding=1),
        )    

    def forward(self, x, mode='denoise'):
        for layer in self.layers:
            if isinstance(layer, (afterConvExpert, LinearExpert)):
                x = layer(x)
            else:
                x = layer(x)
        return x
class midModel(nn.Module):
    def __init__(self):
        super(midModel, self).__init__()
        self.layers = nn.Sequential(
            nn.GroupNorm(16, 256),
            nn.LeakyReLU(),
            ConvExpert(in_channels=256, out_channels=128, kernel_size=3, stride=1 ,padding=1,strange=True), 
            nn.GroupNorm(8, 128),
            nn.LeakyReLU(),
            ConvExpert(in_channels=128, out_channels=256, kernel_size=3, stride=1 ,padding=1,strange=True), 
            nn.GroupNorm(16, 256),
            nn.LeakyReLU(),
        )    

    def forward(self, x, mode='denoise',assign=False):
        for layer in self.layers:
            if isinstance(layer, (ConvExpert, LinearExpert)):
              if(assign):
                x=layer(x,mode,assign=True)
              else:
                x = layer(x, mode)
            else:
                x = layer(x)
        return x
class HidingNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
        self.in_layers = inModel()
        
        self.pre_layers_1 = preModel()
        
        self.pre_layers_2 = preModel()
        
        self.pre_layers_3 = midModel()

        self.mid_secret_path = ConvExpert(in_channels=256, out_channels=128, kernel_size=3, stride=1 ,padding=1,strange=False)
        

        self.mid_cover_path =ConvExpert(in_channels=256, out_channels=128, kernel_size=3, stride=1 ,padding=1)


        self.after_concat_layers_1 = afterModel()
        self.after_concat_layers_2 = afterModel()
        
        self.after_concat_layers_3 = afterModel()
        self.output_layer = outModel()
        
        self.sigmoid = nn.Sigmoid()
    def forward(self, input_1, input_2, task):
        if task == 'hiding':

            secret = input_2
            cover = input_1
            x_s = self.in_layers(secret,'hiding')
            x_s = self.pre_layers_1(x_s,'hiding') + x_s
            x_s = self.pre_layers_2(x_s,'hiding') + x_s
            x_s = self.pre_layers_3(x_s,'hiding') + x_s
            
            x_c = self.in_layers(cover,'hiding',True)
            x_c = self.pre_layers_1(x_c,'hiding',True) + x_c
            x_c = self.pre_layers_2(x_c,'hiding',True) + x_c
            x_c = self.pre_layers_3(x_c,'hiding',True) + x_c
            
            secret_feature = self.mid_secret_path(x_s,'hiding')
            cover_feature = self.mid_cover_path(x_c,'hiding')
            concat_feature = torch.cat((secret_feature, cover_feature), dim=1)

            x = self.after_concat_layers_1(concat_feature,'hiding')
            x = self.after_concat_layers_2(x,'hiding') + x
            x = self.after_concat_layers_3(x,'hiding') + x
            
            stego = self.output_layer(x,'hiding')
            if c.mode == 'test':
                stego = quantization(stego)
            # stego = self.sigmoid(stego)
            out = stego
        elif task == 'recover':

            stego = input_1

            x = self.in_layers(stego,'recover')
            x = self.pre_layers_1(x,'recover') + x
            x = self.pre_layers_2(x,'recover') + x
            x = self.pre_layers_3(x,'recover') + x
            
            stego_feature_1 = self.mid_secret_path(x,'recover')
            stego_feature_2 = self.mid_cover_path(x,'recover')
            concat_feature = torch.cat((stego_feature_1, stego_feature_2), dim=1)

            x = self.after_concat_layers_1(concat_feature,'recover')
            x = self.after_concat_layers_2(x,'recover') + x
            x = self.after_concat_layers_3(x,'recover') + x
            
            secret_rev = self.output_layer(x,'recover')
            if c.mode == 'test':
                secret_rev = quantization(secret_rev)
            # secret_rev = self.sigmoid(secret_rev)
            out = secret_rev
        else: 

            noised_img = input_1
            x = self.in_layers(noised_img)
            x = self.pre_layers_1(x) + x
            x = self.pre_layers_2(x) + x
            x = self.pre_layers_3(x) + x
            
            noised_feature_1 = self.mid_secret_path(x)
            noised_feature_2 = self.mid_cover_path(x)
            concat_feature = torch.cat((noised_feature_1, noised_feature_2), dim=1)

            x = self.after_concat_layers_1(concat_feature)
            x = self.after_concat_layers_2(x) + x
            x = self.after_concat_layers_3(x) + x
       
            noise_residual = self.output_layer(x)
            denoised_img = noised_img - noise_residual
            if c.mode == 'test':
                denoised_img = quantization(denoised_img)
            out = denoised_img

        return out
