import torch
import torchvision.models as tvmodels
from torchsummary import summary
import torchvision.models._utils as _utils
from models.common import SSH, FPN, IntermediateLayerGetterByIndex
import torch.nn as nn
from collections import OrderedDict

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

class IntermediateLayerGetterNested(nn.Module):
    def __init__(self, model, return_layers):
        """
        Args:
            model (nn.Module): the original model
            return_layers (dict): a dict mapping from dotted layer path to user-defined names, 
                                  e.g. {"stage0.conv01": "out1"}
        """
        super().__init__()
        self.model = model
        self.return_layers = return_layers
        self.layer_paths = list(return_layers.keys())

    def _get_from_path(self, model, path):
        for part in path.split('.'):
            model = getattr(model, part)
        return model

    def forward(self, x):
        outputs = OrderedDict()
        modules = {'' : self.model}
        activations = {'' : x}

        # Run forward manually to capture nested layers
        def hook_fn(name):
            def hook(module, input, output):
                outputs[self.return_layers[name]] = output
            return hook

        hooks = []
        for layer_path, alias in self.return_layers.items():
            module = self._get_from_path(self.model, layer_path)
            hooks.append(module.register_forward_hook(hook_fn(layer_path)))

        _ = self.model(x)  # Run full forward once

        # Remove all hooks
        for h in hooks:
            h.remove()

        return outputs
    
from models.backbones import (
    mobilenet_v1_025,
    mobilenet_v1_050,
    mobilenet_v1,
    mobilenet_v2,
    resnet18,
    resnet34,
    resnet50
)

def print_layer_info(model, input_size=(3, 224, 224)):
    """Print detailed layer information"""
    x = torch.randn(1, *input_size)
    model.eval()
    
    def hook_fn(module, input, output):
        print(f"{module.__class__.__name__}: {list(output.shape)}")
    
    # Register hooks
    hooks = []
    for module in model.modules():
        if len(list(module.children())) == 0:  # Only leaf modules
            hooks.append(module.register_forward_hook(hook_fn))
    
    # Forward pass
    with torch.no_grad():
        _ = model(x)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()

if __name__ == "__main__":
  # Method 1: Load ConvNeXt-Tiny model
#   print("Loading CN model...")
  model = tvmodels.convnext_tiny(weights=tvmodels.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)  # or weights=models.ConvNeXt_Tiny_Weights.DEFAULT for newer versions
#   model = tvmodels.convnext_small(weights=tvmodels.ConvNeXt_Small_Weights.IMAGENET1K_V1)  # or weights=models.ConvNeXt_Tiny_Weights.DEFAULT for newer versions
#   model = tvmodels.resnext50_32x4d(weights=tvmodels.ResNeXt50_32X4D_Weights.DEFAULT)    
#   model = resnet34(pretrained=True)

  # 2. Extract the pretrained conv1 weights (original expects 3 input channels)
  original_conv = model.features[0][0]  # Conv2d(3, 96, 4, 4)
  original_weight = original_conv.weight  # Shape: [96, 3, 4, 4]

  # 3. Average the weights over the input channels to simulate grayscale
  new_weight = original_weight.mean(dim=1, keepdim=True)  # Shape: [96, 1, 4, 4]

  # 4. Replace the first conv layer to accept 1 input channel
  model.features[0][0] = nn.Conv2d(
    in_channels=1,
    out_channels=96,
    kernel_size=4,
    stride=4
  )

  # 5. Assign the converted weight
  model.features[0][0].weight.data = new_weight
  model.features[0][0].bias.data = original_conv.bias.data  # keep bias  
  
  dummy_input = torch.randn(1, 1, 48, 48)  # batch of 1 grayscale image
  output = model(dummy_input)
  print(output.shape)
  
  # Print basic model information
  print(f"Model type: {type(model).__name__}")
  print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
  print(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

#   print("\n" + "="*50)
#   print("METHOD 1: Print model architecture")
#   print("="*50)
#   print(model)

#   print("\n" + "="*50)
#   print("METHOD 2: Print named modules")
#   print("="*50)
#   for name, module in model.named_modules():
#       if len(list(module.children())) == 0:  # Only leaf modules
#           print(f"{name}: {module}")

#   print("\n" + "="*50)
#   print("METHOD 3: Print layer names and shapes")
#   print("="*50)
#   print_layer_info(model)

#   print("="*50)
  # yy = _utils.IntermediateLayerGetter(model, {'features': 1, 'features.5': 2, 'features.7': 3})
#   getter = _utils.IntermediateLayerGetter(model, {'layer2': 1, 'layer3': 2, 'layer4': 3})
  # yy = IntermediateLayerGetterByIndex(model, {})
  # out = yy(torch.rand(1, 3, 224, 224)) 

#   getter = IntermediateLayerGetterNested(
#     model,
#     return_layers={
#         "features.3": 1,
#         "features.5": 2,
#         "features.7": 3
#     }
#   )

#   output = getter(torch.rand(1, 3, 800, 1200))

#   print([(k, v.shape) for k, v in output.items()]) 
#   print(output.keys())
#   print(out)