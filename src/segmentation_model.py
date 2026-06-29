import segmentation_models_pytorch as smp

def get_model(model_name="unet", encoder_name="resnet34", num_classes=4, encoder_weights="imagenet"):
    """
    Helper function to instantiate U-Net or DeepLabV3+ models with SMP.
    
    Args:
        model_name (str): 'unet' or 'deeplabv3+'
        encoder_name (str): name of the encoder backbone, e.g. 'resnet34', 'resnet50'
        num_classes (int): number of segmentation output channels
        encoder_weights (str): pre-trained weights to load, e.g. 'imagenet'
        
    Returns:
        torch.nn.Module: instantiated PyTorch segmentation model
    """
    model_name = model_name.lower().strip()
    
    if model_name == "unet":
        print(f"Creating U-Net model with {encoder_name} encoder...")
        model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=num_classes
        )
    elif model_name == "deeplabv3+" or model_name == "deeplabv3plus":
        print(f"Creating DeepLabV3+ model with {encoder_name} encoder...")
        model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=num_classes
        )
    else:
        raise ValueError(f"Unsupported model name: {model_name}. Choose 'unet' or 'deeplabv3+'")
        
    return model
