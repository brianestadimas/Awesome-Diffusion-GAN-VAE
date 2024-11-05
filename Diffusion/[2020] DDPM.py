from diffusers import DDPMPipeline
from PIL import Image
import torch

class DDPM:
    def __init__(self, model_id, device=None):
        """
        Initialize the DDPM model with a specified model ID and device, and load the pipeline.
        
        Parameters:
        - model_id (str): The ID of the pretrained model to load (from Hugging Face's model hub).
        - device (str): Device to run the model on ("cuda" for GPU, "cpu" for CPU).
        """
        
        # Example model IDs (for reference):
        # model_id = "fusing/ddpm-lsun-bedroom-ema"
        # model_id = "google/ddpm-cifar10-32"
        # model_id = "google/ddpm-celebahq-256"
        # model_id = "google/ddpm-ema-church-256"
        # model_id = "fusing/ddpm-celeba-hq-ema"
    
        self.model_id = model_id
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.pipeline = DDPMPipeline.from_pretrained(self.model_id)
        self.pipeline.to(self.device)

    def generate_image(self, seed=None):
        """
        Generate an image using the loaded DDPM pipeline.
        
        Parameters:
        - seed (int): Random seed for reproducibility (optional).
        
        Returns:
        - image (PIL.Image): The generated image.
        """

        generator = torch.manual_seed(seed) if seed else None

        with torch.no_grad():
            images = self.pipeline(batch_size=1, generator=generator).images
            image = images[0]

        return image
