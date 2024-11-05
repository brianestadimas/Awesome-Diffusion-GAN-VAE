from diffusers import DDIMPipeline
from PIL import Image
import torch

class DDIM:
    def __init__(self, model_id, device=None):
        """
        Initialize the DDIM model with a specified model ID and device, and load the pipeline.
        
        Parameters:
        - model_id (str): The ID of the pretrained model to load (from Hugging Face's model hub).
        - device (str): Device to run the model on ("cuda" for GPU, "cpu" for CPU).
        
        Example model IDs (for reference):
        - model_id = "fusing/ddim-celeba-hq"
        - model_id = "fusing/ddim-lsun-bedroom"
        - model_id = "fusing/ddim-lsun-church"
        - model_id = "krasnova/ddim_afhq_64"
        - model_id = "alibidaran/ddim-celebahq-finetuned-oldimages-2epochs"
        """
        self.model_id = model_id
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.pipeline = DDIMPipeline.from_pretrained(self.model_id)
        self.pipeline.to(self.device)

    def generate_image(self, seed=None):
        """
        Generate an image using the loaded DDIM pipeline.
        
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
