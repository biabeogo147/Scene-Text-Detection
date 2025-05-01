import argparse

import torch
import os
from torchvision import transforms
from PIL import Image, ImageDraw
from recognition.network import CRNN


def load_crnn_model(model_path, vocab, hidden_size, n_layers, device):
    vocab_size = len(vocab)
    char_to_idx = {char: idx + 1 for idx, char in enumerate(sorted(vocab))}
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}

    model = CRNN(vocab_size, hidden_size, n_layers)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    return model, idx_to_char


def recognize_text(image_path, model, idx_to_char, device):
    preprocess = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((100, 420)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    img = Image.open(image_path).convert("RGB")
    img_tensor = preprocess(img).unsqueeze(0).to(device)
    
    # Perform inference
    with torch.no_grad():
        logits = model(img_tensor)
    decoded_text = decode(logits.permute(1, 0, 2).argmax(2), idx_to_char)[0]
    return decoded_text


def decode(encoded_sequences, idx_to_char, blank_char="-"):
    decoded_sequences = []

    for seq in encoded_sequences:
        decoded_label = []
        prev_char = None  # To track the previous character

        for token in seq:
            if token != 0:  # Ignore padding (token = 0)
                char = idx_to_char[token.item()]
                # Append the character if it's not a blank or the same as the previous character
                if char != blank_char:
                    if char != prev_char or prev_char == blank_char:
                        decoded_label.append(char)
                prev_char = char  # Update previous character

        decoded_sequences.append("".join(decoded_label))

    return decoded_sequences


def visualize(image_path, recognized_text, output_folder=None):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    # Draw label on the image
    text_position = (10, 10)  # Top-left corner
    draw.rectangle(
        [text_position, (text_position[0] + len(recognized_text) * 10, text_position[1] + 25)],
        fill="black"
    )
    draw.text(text_position, recognized_text, fill="white")

    # Save the result
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, os.path.basename(image_path))
        img.save(output_path)
        print(f"Saved visualization to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for text detection and recognition")
    parser.add_argument("--im_path", type=str, required=True, help="Path to the input image")
    args = parser.parse_args()

    model_path = "../models/ocr_crnn.pt"
    image_path = args.im_path
    vocab = "0123456789abcdefghijklmnopqrstuvwxyz-" 
    save_dir = "results"

    hidden_size = 256
    n_layers = 3
    device = "cuda" if torch.cuda.is_available() else "cpu"

    crnn_model, idx_to_char = load_crnn_model(model_path, vocab, hidden_size, n_layers, device)

    recognized_text = recognize_text(image_path, crnn_model, idx_to_char, device)
    print(f"Recognized Text: {recognized_text}")

    visualize(image_path, recognized_text, save_dir)
