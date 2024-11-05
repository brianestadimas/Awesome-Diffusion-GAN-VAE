from diffusers import DiffusionPipeline
from PIL import Image
import torch

class LDM:
    def __init__(self, model_id, device=None):
        """
        Initialize the LDM model with a specified model ID and device, and load the pipeline.
        
        Parameters:
        - model_id (str): The ID of the pretrained model to load (from Hugging Face's model hub).
        - device (str): Device to run the model on ("cuda" for GPU, "cpu" for CPU).
        
        Example model IDs (for reference):
        - model_id = "CompVis/ldm-text2im-large-256"
        - model_id = "CompVis/ldm-celebahq-256"
        - model_id = "ndbao2002/ldm-cifar-32"
        - model_id = "kaayaanil/ldm-ffhq-256"
        """
        self.model_id = model_id
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.pipeline = DiffusionPipeline.from_pretrained(self.model_id)
        self.pipeline.to(self.device)

    def generate_image(self, num_inference_steps=200, seed=None):
        """
        Generate an image using the loaded LDM pipeline.
        
        Parameters:
        - num_inference_steps (int): Number of denoising steps (higher values produce higher quality images).
        - seed (int): Random seed for reproducibility (optional).
        
        Returns:
        - image (PIL.Image): The generated image.
        """
        generator = torch.manual_seed(seed) if seed else None

        with torch.no_grad():
            images = self.pipeline(num_inference_steps=num_inference_steps, generator=generator)["sample"]
            image = images[0]

        return image