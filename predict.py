import argparse
import torch
from torchvision import models
from PIL import Image
import json
from torchvision import transforms

def get_input_args():
    """
    Parse command-line arguments for prediction.
    """
    parser = argparse.ArgumentParser(description="Predict image class using a trained model.")
    parser.add_argument('input', type=str, help='Path to input image.')
    parser.add_argument('checkpoint', type=str, help='Path to model checkpoint.')
    parser.add_argument('--top_k', type=int, default=5, help='Return top K predictions.')
    parser.add_argument('--category_names', type=str, help='Path to JSON file mapping categories to names.')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference.')
    return parser.parse_args()

def load_checkpoint(filepath):
    """
    Load a model checkpoint and rebuild the model.
    """
    checkpoint = torch.load(filepath, map_location='cuda' if torch.cuda.is_available() else 'cpu')
    if checkpoint['arch'] == 'VGG':
        model = models.vgg16(pretrained=True)
    elif checkpoint['arch'] == 'ResNet':
        model = models.resnet18(pretrained=True)
    else:
        raise ValueError(f"Unsupported architecture: {checkpoint['arch']}")
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    # Ensure model is in evaluation mode
    model.eval()
    
    return model

def process_image(image_path):
    """
    Process an image to meet the input requirements of the model.
    """
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Open the image and apply transforms
    image = Image.open(image_path)
    image = transform(image)
    
    return image

def predict(image_path, model, top_k, device):
    """
    Predict the class (or classes) of an image using a trained model.
    """
    # Process the image
    image = process_image(image_path).unsqueeze(0).to(device)  # Add batch dimension
    
    # Move model to the specified device
    model.to(device)
    
    # Make predictions
    with torch.no_grad():
        log_ps = model(image)
        ps = torch.exp(log_ps)
    
    # Get the top K probabilities and indices
    top_probs, top_indices = ps.topk(top_k, dim=1)
    top_probs = top_probs.cpu().numpy()[0]
    top_indices = top_indices.cpu().numpy()[0]
    
    # Map indices to classes
    idx_to_class = {idx: cls for cls, idx in model.class_to_idx.items()}
    top_classes = [idx_to_class[idx] for idx in top_indices]
    
    return top_probs, top_classes

def main():
    # Parse arguments
    args = get_input_args()
    
    # Load the model checkpoint
    model = load_checkpoint(args.checkpoint)
    
    # Set the device
    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
    
    # Predict the class
    probs, classes = predict(args.input, model, args.top_k, device)
    
    # Map categories to names if provided
    if args.category_names:
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
        class_names = [cat_to_name[str(cls)] for cls in classes]
    else:
        class_names = classes
    
    # Print results
    print(f"Top {args.top_k} Predictions:")
    for prob, name in zip(probs, class_names):
        print(f"{name}: {prob:.4f}")

if __name__ == "__main__":
    main()
