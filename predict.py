from cog import BasePredictor, Input
from PIL import Image
import torch
import depth_pro

class Predictor(BasePredictor):
    def setup(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Load the model and transforms
        self.model, self.transform = depth_pro.create_model_and_transforms()
        self.model.to(self.device)
        self.model.eval()

    def predict(
        self,
        image_path: str = Input(description="Path to the input image"),
    ) -> dict:
        # Load and preprocess the image
        image, _, f_px = depth_pro.load_rgb(image_path)
        image = self.transform(image).unsqueeze(0).to(self.device)

        # Run inference
        with torch.no_grad():
            prediction = self.model.infer(image, f_px=f_px)

        depth_map = prediction["depth"].cpu().numpy()
        focal_length_px = prediction["focallength_px"]

        # Return results as dictionary
        return {
            "depth_map": depth_map.tolist(),
            "focal_length_px": focal_length_px
        }
