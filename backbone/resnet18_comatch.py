#From https://pytorch.org/vision/0.8/_modules/torchvision/models/resnet.html

import torch
import torch.nn as nn
from typing import Dict, Optional, Any
import os

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)



class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        
    
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

    
    
class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4
    
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

    
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
            
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        #For COMATCH
        self.proj = True
        
        if self.proj:
            self.l2norm = Normalize(2)
            self.fc1 = nn.Linear(512, 512)
            self.relu_mlp = nn.LeakyReLU(inplace=True, negative_slope=0.1)
            self.fc2 = nn.Linear(512, 64)
            
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
                
        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
                    
                    
    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
            
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
    
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))
            
        return nn.Sequential(*layers)
    
    
    def _forward_impl(self, x):
        # See note [TorchScript super()]
        feat = self.conv1(x)
        feat = self.bn1(feat)
        feat = self.relu(feat)
        feat = self.maxpool(feat)

        feat = self.layer1(feat)
        feat = self.layer2(feat)
        feat = self.layer3(feat)
        feat = self.layer4(feat)

        feat = self.avgpool(feat)
        feat = torch.flatten(feat, 1)
        out = self.fc(feat)
        
        if self.proj:
            feat = self.fc1(feat)
            feat = self.relu_mlp(feat)       
            feat = self.fc2(feat)

            feat = self.l2norm(feat)  
            return out, feat
            
        else:    
            return out
        
    
    def forward(self, x):
        return self._forward_impl(x)
    
    
    
def _resnet(arch, block, layers, num_classes, pretrained, progress, **kwargs):
    model = ResNet(block, layers, num_classes, **kwargs)
    if pretrained:
        print('!!!!!!!!!!!!!!!!!!!Using Pretrain!!!!!!!!!!!!!')
        
        state_dict = torch.load('/cluster/tufts/hugheslab/zhuang12/SemiSelfEvaluationProject/SemiSelfEvaluation/Semi-supervised/resnet18-5c106cde.pth')
#         state_dict = load_state_dict_from_url(model_urls[arch],
#                                               progress=progress)

        # Define the keys that you do not want to load into the model.
        keys_to_delete = ["fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias", "fc.weight", "fc.bias"]
        for key in keys_to_delete:
            if key in state_dict:
                del state_dict[key]
    
        model.load_state_dict(state_dict, strict=False)
    return model


class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out
    
    
def build_resnet18(num_classes, pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], num_classes, pretrained, progress,
                   **kwargs)
 
# def get_dir():
#     r"""
#     Get the Torch Hub cache directory used for storing downloaded models & weights.

#     If :func:`~torch.hub.set_dir` is not called, default path is ``$TORCH_HOME/hub`` where
#     environment variable ``$TORCH_HOME`` defaults to ``$XDG_CACHE_HOME/torch``.
#     ``$XDG_CACHE_HOME`` follows the X Design Group specification of the Linux
#     filesystem layout, with a default value ``~/.cache`` if the environment
#     variable is not set.
#     """
#     # Issue warning to move data if old env is set
#     if os.getenv('TORCH_HUB'):
#         warnings.warn('TORCH_HUB is deprecated, please use env TORCH_HOME instead')

#     if _hub_dir is not None:
#         return _hub_dir
#     return os.path.join(_get_torch_home(), 'hub')

# #https://github.com/pytorch/pytorch/blob/f15af1987777db2cb590aab98cefe74bfe086b48/torch/hub.py#L689
# # def load_state_dict_from_url(
# #     url: str,
# #     model_dir: Optional[str] = None,
# #     map_location: MAP_LOCATION = None,
# #     progress: bool = True,
# #     check_hash: bool = False,
# #     file_name: Optional[str] = None,
# #     weights_only: bool = False,
# # ) -> Dict[str, Any]:
# def load_state_dict_from_url(
#     url: str,
#     model_dir= None,
#     map_location= None,
#     progress= True,
#     check_hash= False,
#     file_name= None,
#     weights_only: bool = False,
# ) -> Dict[str, Any]:
#     r"""Loads the Torch serialized object at the given URL.

#     If downloaded file is a zip file, it will be automatically
#     decompressed.

#     If the object is already present in `model_dir`, it's deserialized and
#     returned.
#     The default value of ``model_dir`` is ``<hub_dir>/checkpoints`` where
#     ``hub_dir`` is the directory returned by :func:`~torch.hub.get_dir`.

#     Args:
#         url (str): URL of the object to download
#         model_dir (str, optional): directory in which to save the object
#         map_location (optional): a function or a dict specifying how to remap storage locations (see torch.load)
#         progress (bool, optional): whether or not to display a progress bar to stderr.
#             Default: True
#         check_hash(bool, optional): If True, the filename part of the URL should follow the naming convention
#             ``filename-<sha256>.ext`` where ``<sha256>`` is the first eight or more
#             digits of the SHA256 hash of the contents of the file. The hash is used to
#             ensure unique names and to verify the contents of the file.
#             Default: False
#         file_name (str, optional): name for the downloaded file. Filename from ``url`` will be used if not set.
#         weights_only(bool, optional): If True, only weights will be loaded and no complex pickled objects.
#             Recommended for untrusted sources. See :func:`~torch.load` for more details.

#     Example:
#         >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_HUB)
#         >>> state_dict = torch.hub.load_state_dict_from_url('https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth')

#     """
#     # Issue warning to move data if old env is set
#     if os.getenv('TORCH_MODEL_ZOO'):
#         warnings.warn('TORCH_MODEL_ZOO is deprecated, please use env TORCH_HOME instead')

#     if model_dir is None:
#         hub_dir = get_dir()
#         model_dir = os.path.join(hub_dir, 'checkpoints')

#     try:
#         os.makedirs(model_dir)
#     except OSError as e:
#         if e.errno == errno.EEXIST:
#             # Directory already exists, ignore.
#             pass
#         else:
#             # Unexpected OSError, re-raise.
#             raise

#     parts = urlparse(url)
#     filename = os.path.basename(parts.path)
#     if file_name is not None:
#         filename = file_name
#     cached_file = os.path.join(model_dir, filename)
#     if not os.path.exists(cached_file):
#         sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
#         hash_prefix = None
#         if check_hash:
#             r = HASH_REGEX.search(filename)  # r is Optional[Match[str]]
#             hash_prefix = r.group(1) if r else None
#         download_url_to_file(url, cached_file, hash_prefix, progress=progress)

#     if _is_legacy_zip_format(cached_file):
#         return _legacy_zip_load(cached_file, model_dir, map_location, weights_only)
#     return torch.load(cached_file, map_location=map_location, weights_only=weights_only)  
    
    
    
    
    