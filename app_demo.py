import gradio as gr
import torch

from transformers import ViTFeatureExtractor, ViTForImageClassification

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ViTForImageClassification.from_pretrained("vit-base-beans/checkpoint-200").to(device)
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')

labels = ['angular_leaf_spot', 'bean_rust', 'healthy']

def classify(image):
  """
  It takes an image, extracts features from it, feeds those features into the
  model, and returns a dictionary of labels and their corresponding confidence
  scores
  
  :param image: the image to classify
  :return: A dictionary of labels and their corresponding probabilities.
  """
  features = feature_extractor(image, return_tensors='pt').to(device)
  logits = model(features["pixel_values"])[-1]
  probability = torch.nn.functional.softmax(logits, dim=-1)
  probs = probability[0].detach().cpu().numpy()
  confidences = {label: float(probs[i]) for i, label in enumerate(labels)} 
  return confidences

interface = gr.Interface(
    fn=classify, 
    inputs="image", 
    outputs=gr.Label(num_top_classes=1),
    examples=["example1.png", "example2.png"]
    )

interface.launch()