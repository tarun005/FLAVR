# Sourced from https://github.com/myungsub/CAIN/blob/master/loss.py, who sourced from https://github.com/thstkdgus35/EDSR-PyTorch/tree/master/src/loss
# Added Huber loss in addition.


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import pytorch_msssim

class MeanShift(nn.Conv2d):
    def __init__(self, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False

class HuberLoss(nn.Module):

    def __init__(self , delta=1):

        super().__init__()
        self.delta = delta

    def forward(self , sr , hr):

        l1 = torch.abs(sr - hr)
        mask = l1<self.delta

        sq_loss = .5*(l1**2) 
        abs_loss = self.delta*(l1 - .5*self.delta)

        return torch.mean(mask*sq_loss + (~mask)*(abs_loss))


class VGG(nn.Module):
    def __init__(self, loss_type):
        super(VGG, self).__init__()
        vgg_features = models.vgg19(pretrained=True).features
        modules = [m for m in vgg_features]
        conv_index = loss_type[-2:]
        if conv_index == '22':
            self.vgg = nn.Sequential(*modules[:8])
        elif conv_index == '33':
            self.vgg = nn.Sequential(*modules[:16])
        elif conv_index == '44':
            self.vgg = nn.Sequential(*modules[:26])
        elif conv_index == '54':
            self.vgg = nn.Sequential(*modules[:35])
        elif conv_index == 'P':
            self.vgg = nn.ModuleList([
                nn.Sequential(*modules[:8]),
                nn.Sequential(*modules[8:16]),
                nn.Sequential(*modules[16:26]),
                nn.Sequential(*modules[26:35])
            ])
        self.vgg = nn.DataParallel(self.vgg).cuda()

        vgg_mean = (0.485, 0.456, 0.406)
        vgg_std = (0.229, 0.224, 0.225)
        self.sub_mean = MeanShift(vgg_mean, vgg_std)
        self.vgg.requires_grad = False
        # self.criterion = nn.L1Loss()
        self.conv_index = conv_index

    def forward(self, sr, hr):
        def _forward(x):
            x = self.sub_mean(x)
            x = self.vgg(x)
            return x
        def _forward_all(x):
            feats = []
            x = self.sub_mean(x)
            for module in self.vgg.module:
                x = module(x)
                feats.append(x)
            return feats

        if self.conv_index == 'P':
            vgg_sr_feats = _forward_all(sr)
            with torch.no_grad():
                vgg_hr_feats = _forward_all(hr.detach())
            loss = 0
            for i in range(len(vgg_sr_feats)):
                loss_f = F.mse_loss(vgg_sr_feats[i], vgg_hr_feats[i])
                #print(loss_f)
                loss += loss_f
            #print()
        else:
            vgg_sr = _forward(sr)
            with torch.no_grad():
                vgg_hr = _forward(hr.detach())
            loss = F.mse_loss(vgg_sr, vgg_hr)

        return loss


# For Adversarial loss
class BasicBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=False, bn=True, act=nn.ReLU(True)):
        m = [nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size//2), stride=stride, bias=bias)]
        if bn: m.append(nn.BatchNorm2d(out_channels))
        if act is not None: m.append(act)
        super(BasicBlock, self).__init__(*m)

class Discriminator(nn.Module):
    def __init__(self, args, gan_type='GAN'):
        super(Discriminator, self).__init__()

        in_channels = 3
        out_channels = 64
        depth = 7
        #bn = not gan_type == 'WGAN_GP'
        bn = True
        act = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        m_features = [
            BasicBlock(in_channels, out_channels, 3, bn=bn, act=act)
        ]
        for i in range(depth):
            in_channels = out_channels
            if i % 2 == 1:
                stride = 1
                out_channels *= 2
            else:
                stride = 2
            m_features.append(BasicBlock(
                in_channels, out_channels, 3, stride=stride, bn=bn, act=act
            ))

        self.features = nn.Sequential(*m_features)

        self.patch_size = args.patch_size
        feature_patch_size = self.patch_size // (2**((depth + 1) // 2))
        #patch_size = 256 // (2**((depth + 1) // 2))
        m_classifier = [
            nn.Linear(out_channels * feature_patch_size**2, 1024),
            act,
            nn.Linear(1024, 1)
        ]
        self.classifier = nn.Sequential(*m_classifier)

    def forward(self, x):
        if x.size(2) != self.patch_size or x.size(3) != self.patch_size:
            midH, midW = x.size(2) // 2, x.size(3) // 2
            p = self.patch_size // 2
            x = x[:, :, (midH - p):(midH - p + self.patch_size), (midW - p):(midW - p + self.patch_size)]
        features = self.features(x)
        output = self.classifier(features.view(features.size(0), -1))

        return output


import torch.optim as optim
class Adversarial(nn.Module):
    def __init__(self, args, gan_type):
        super(Adversarial, self).__init__()
        self.gan_type = gan_type
        self.gan_k = 1 #args.gan_k
        self.discriminator = torch.nn.DataParallel(Discriminator(args, gan_type))
        if gan_type != 'WGAN_GP':
            self.optimizer = optim.Adam(
                self.discriminator.parameters(),
                betas=(0.9, 0.99), eps=1e-8, lr=1e-4
            )
        else:
            self.optimizer = optim.Adam(
                self.discriminator.parameters(),
                betas=(0, 0.9), eps=1e-8, lr=1e-5
            )
        # self.scheduler = utility.make_scheduler(args, self.optimizer)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    def forward(self, fake, real, fake_input0=None, fake_input1=None, fake_input_mean=None):
        # def forward(self, fake, real):
        fake_detach = fake.detach()
        if fake_input0 is not None:
            fake0, fake1 = fake_input0.detach(), fake_input1.detach()
        if fake_input_mean is not None:
            fake_m = fake_input_mean.detach()
        # print(fake.size(), fake_input0.size(), fake_input1.size(), fake_input_mean.size())

        self.loss = 0
        for _ in range(self.gan_k):
            self.optimizer.zero_grad()
            d_fake = self.discriminator(fake_detach)

            if fake_input0 is not None and fake_input1 is not None:
                d_fake0 = self.discriminator(fake0)
                d_fake1 = self.discriminator(fake1)
            if fake_input_mean is not None:
                d_fake_m = self.discriminator(fake_m)
            
            # print(d_fake.size(), d_fake0.size(), d_fake1.size(), d_fake_m.size())
            
            d_real = self.discriminator(real)
            if self.gan_type == 'GAN':
                label_fake = torch.zeros_like(d_fake)
                label_real = torch.ones_like(d_real)
                loss_d \
                    = F.binary_cross_entropy_with_logits(d_fake, label_fake) \
                        + F.binary_cross_entropy_with_logits(d_real, label_real)
                if fake_input0 is not None and fake_input1 is not None:
                    loss_d += F.binary_cross_entropy_with_logits(d_fake0, label_fake) \
                        + F.binary_cross_entropy_with_logits(d_fake1, label_fake)
                if fake_input_mean is not None:
                    loss_d += F.binary_cross_entropy_with_logits(d_fake_m, label_fake)

            elif self.gan_type.find('WGAN') >= 0:
                loss_d = (d_fake - d_real).mean()
                if self.gan_type.find('GP') >= 0:
                    epsilon = torch.rand_like(fake).view(-1, 1, 1, 1)
                    hat = fake_detach.mul(1 - epsilon) + real.mul(epsilon)
                    hat.requires_grad = True
                    d_hat = self.discriminator(hat)
                    gradients = torch.autograd.grad(
                        outputs=d_hat.sum(), inputs=hat,
                        retain_graph=True, create_graph=True, only_inputs=True
                    )[0]
                    gradients = gradients.view(gradients.size(0), -1)
                    gradient_norm = gradients.norm(2, dim=1)
                    gradient_penalty = 10 * gradient_norm.sub(1).pow(2).mean()
                    loss_d += gradient_penalty

            # Discriminator update
            self.loss += loss_d.item()
            if self.training:
                loss_d.backward()
                self.optimizer.step()

            if self.gan_type == 'WGAN':
                for p in self.discriminator.parameters():
                    p.data.clamp_(-1, 1)

        self.loss /= self.gan_k

        d_fake_for_g = self.discriminator(fake)
        if self.gan_type == 'GAN':
            loss_g = F.binary_cross_entropy_with_logits(
                d_fake_for_g, label_real
            )
        elif self.gan_type.find('WGAN') >= 0:
            loss_g = -d_fake_for_g.mean()

        # Generator loss
        return loss_g

    def state_dict(self, *args, **kwargs):
        state_discriminator = self.discriminator.state_dict(*args, **kwargs)
        state_optimizer = self.optimizer.state_dict()

        return dict(**state_discriminator, **state_optimizer)


# Some references
# https://github.com/kuc2477/pytorch-wgan-gp/blob/master/model.py
# OR
# https://github.com/caogang/wgan-gp/blob/master/gan_cifar10.py


# Wrapper of loss functions
class Loss(nn.modules.loss._Loss):
    def __init__(self, args):
        super(Loss, self).__init__()
        print('Preparing loss function:')

        self.loss = []
        self.loss_module = nn.ModuleList()
        for loss in args.loss.split('+'):
            weight, loss_type = loss.split('*')
            if loss_type == 'MSE':
                loss_function = nn.MSELoss()
            elif loss_type == 'Huber':
                loss_function = HuberLoss(delta=.5)
            elif loss_type == 'L1':
                loss_function = nn.L1Loss()
            elif loss_type.find('VGG') >= 0:
                loss_function = VGG(loss_type[3:])
            elif loss_type == 'SSIM':
                loss_function = pytorch_msssim.SSIM(val_range=1.)
            elif loss_type.find('GAN') >= 0:
                loss_function = Adversarial(args, loss_type)

            self.loss.append({
                'type': loss_type,
                'weight': float(weight),
                'function': loss_function}
            )
            if loss_type.find('GAN') >= 0 >= 0:
                self.loss.append({'type': 'DIS', 'weight': 1, 'function': None})

        if len(self.loss) > 1:
            self.loss.append({'type': 'Total', 'weight': 0, 'function': None})

        for l in self.loss:
            if l['function'] is not None:
                print('{:.3f} * {}'.format(l['weight'], l['type']))
                self.loss_module.append(l['function'])

        device = torch.device('cuda' if args.cuda else 'cpu')
        self.loss_module.to(device)
        #if args.precision == 'half': self.loss_module.half()
        if args.cuda:# and args.n_GPUs > 1:
            self.loss_module = nn.DataParallel(self.loss_module)


    def forward(self, sr, hr, fake_imgs=None):
        loss = 0
        losses = {}
        for i, l in enumerate(self.loss):
            if l['function'] is not None:
                if l['type'] == 'GAN':
                    if fake_imgs is None:
                        fake_imgs = [None, None, None]
                    _loss = l['function'](sr, hr, fake_imgs[0], fake_imgs[1], fake_imgs[2])
                else:
                    _loss = l['function'](sr, hr)
                effective_loss = l['weight'] * _loss
                losses[l['type']] = effective_loss
                loss += effective_loss
            elif l['type'] == 'DIS':
                losses[l['type']] = self.loss[i - 1]['function'].loss


        return loss, losses
