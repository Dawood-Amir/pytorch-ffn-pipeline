import torch
import numpy as np
import config as config
from model.ffn import IrisPyTorchFFN

def run_production_inference():
    print(f" Initializing modern production inference pipeline on: {config.DEVICE}")

    #  Reconstruct model backbone layout blueprint
    model = IrisPyTorchFFN(
        input_dim=config.INPUT_DIM, 
        hidden_dim=config.HIDDEN_DIM, 
        output_dim=config.OUTPUT_DIM,
        dropout_rate=config.DROPOUT_RATE
    ).to(config.DEVICE)

    # De-serialize and bind saved modern model parameter weights
    try:
        model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE))
        print(" Pretrained modern model weights successfully restored.")
    except FileNotFoundError:
        print(f" Error: Could not find checkpoint file at '{config.MODEL_SAVE_PATH}'. Run train.py first.")
        return

    model.eval()  

    # Create a mock fresh sample (Simulating real-time incoming user data)
    unseen_sample = np.array([[0.5, -1.2, 0.8, 1.4]], dtype=np.float32)
    sample_tensor = torch.from_numpy(unseen_sample).to(config.DEVICE)

    #  Compute predictions
    with torch.no_grad():
        raw_logits = model(sample_tensor)
        probabilities = torch.softmax(raw_logits, dim=1)
        confidence, predicted_class = torch.max(probabilities, dim=1)

    # Map indexes back to string tags
    class_mapping = {0: "Iris-Setosa", 1: "Iris-Versicolor", 2: "Iris-Virginica"}
    result_tag = class_mapping[predicted_class.item()]

    print("\n--- Inference Telemetry Output ---")
    print(f" Raw Probabilities Distribution Matrix: {probabilities.cpu().numpy()[0]}")
    print(f" Classification Result: {result_tag} ({confidence.item() * 100:.2f}% Certainty Mapping)")

if __name__ == "__main__":
    run_production_inference()