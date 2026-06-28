import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import gradio as gr
import numpy as np
import cv2
import tensorflow as tf

def get_mapping():
    mapping = {}
    for i in range(10): mapping[i] = str(i)
    return mapping

class_mapping = get_mapping()

def load_model():
    if not os.path.exists('mnist_model.h5'):
        return None
    return tf.keras.models.load_model('mnist_model.h5')

model = load_model()

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """Generates Grad-CAM heatmap."""
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if isinstance(preds, list):
            preds = preds[0]
        if isinstance(last_conv_layer_output, list):
            last_conv_layer_output = last_conv_layer_output[0]
            
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.math.reduce_max(heatmap)
    if max_val != 0:
        heatmap = heatmap / max_val
    return heatmap.numpy()

import traceback

def predict_and_explain(img_dict):
    try:
        if model is None:
            return "Model not found. Run 'train_model.py' first.", None
            
        if isinstance(img_dict, dict):
            img = img_dict.get('composite', img_dict.get('image', img_dict.get('background')))
        else:
            img = img_dict
            
        if img is None:
            return "Please draw a number.", None
            
        if len(img.shape) == 3:
            if img.shape[2] == 4 and np.min(img[:,:,3]) < 255:
                # Gradio Sketchpad transparent background; use alpha channel as grayscale
                img_gray = img[:,:,3]
            else:
                img_gray = cv2.cvtColor(img[:,:,:3], cv2.COLOR_RGB2GRAY)
                if np.mean(img_gray) > 127:
                    img_gray = 255 - img_gray
        else:
            img_gray = img
            if np.mean(img_gray) > 127:
                img_gray = 255 - img_gray
            
        coords = cv2.findNonZero(img_gray)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            crop = img_gray[y:y+h, x:x+w]
            
            # Preserve aspect ratio by placing the crop in a square canvas
            max_side = max(w, h)
            margin = int(max_side * 0.25)
            target_size = max_side + 2 * margin
            
            square_img = np.zeros((target_size, target_size), dtype=np.uint8)
            
            start_y = margin + (max_side - h) // 2
            start_x = margin + (max_side - w) // 2
            
            square_img[start_y:start_y+h, start_x:start_x+w] = crop
            img_gray = square_img
            
        img_resized = cv2.resize(img_gray, (28, 28), interpolation=cv2.INTER_AREA)
        img_normalized = img_resized.astype('float32') / 255.0
        
        img_input = np.expand_dims(np.expand_dims(img_normalized, axis=0), axis=-1)
        
        preds = model.predict(img_input)[0]
        pred_index = np.argmax(preds)
        confidence = preds[pred_index]
        predicted_char = class_mapping[pred_index]
        
        result_text = f"Predicted Number: **{predicted_char}**\nConfidence: {confidence*100:.2f}%"
        
        conv_layers = [layer.name for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)]
        if not conv_layers:
            return result_text, None
            
        last_conv_layer_name = conv_layers[-1]
        heatmap = make_gradcam_heatmap(img_input, model, last_conv_layer_name, pred_index)
        
        heatmap = cv2.resize(heatmap, (28, 28))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        img_color = cv2.cvtColor((img_normalized * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        superimposed_img = heatmap * 0.4 + img_color
        superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
        
        superimposed_img = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)
        
        # Upscale the final 28x28 heatmap image so it doesn't squish the UI
        superimposed_img = cv2.resize(superimposed_img, (400, 400), interpolation=cv2.INTER_NEAREST)
        
        return result_text, superimposed_img
    except Exception as e:
        return f"CRASH: {str(e)}\n```python\n{traceback.format_exc()}\n```", None

custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,600;1,600&display=swap');

body {
    background-image: url("https://www.transparenttextures.com/patterns/old-wall.png") !important;
}

h1 { 
    text-align: center; 
    font-family: 'Playfair Display', serif !important;
    color: #4a3c31 !important; 
    font-size: 3.5em !important; 
    margin-bottom: 10px !important; 
    text-shadow: 1px 1px 2px #d3c5b3;
    border-bottom: 2px solid #8b7355;
    padding-bottom: 15px;
    background: none !important;
    -webkit-text-fill-color: #4a3c31 !important;
}

.subtitle { 
    text-align: center; 
    font-size: 1.3em; 
    color: #5c4e40 !important; 
    margin-bottom: 30px; 
    font-style: italic;
}
button.primary, button.secondary {
    box-shadow: 4px 4px 0px #8b7355 !important;
    transition: all 0.1s !important;
}

button.primary:hover, button.secondary:hover {
    transform: translate(2px, 2px) !important;
    box-shadow: 2px 2px 0px #8b7355 !important;
}

/* Brute-force fix for invisible Gradio Sketchpad toolbars */
button, .icon-button, [aria-label] {
    color: #3e3222 !important;
}
button svg, .icon-button svg, .gradio-container svg {
    color: #3e3222 !important;
}
"""

vintage_theme = gr.themes.Soft(
    primary_hue="amber",
    neutral_hue="stone",
    font=[gr.themes.GoogleFont("Courier Prime"), "monospace"]
).set(
    body_background_fill="#f4ecd8",
    body_background_fill_dark="#f4ecd8",
    body_text_color="#3e3222",
    body_text_color_dark="#3e3222",
    body_text_color_subdued="#5c4e40",
    body_text_color_subdued_dark="#5c4e40",
    background_fill_primary="#fdfaf4",
    background_fill_primary_dark="#fdfaf4",
    background_fill_secondary="#faeedd",
    background_fill_secondary_dark="#faeedd",
    border_color_primary="#8b7355",
    border_color_primary_dark="#8b7355",
    block_background_fill="#fdfaf4",
    block_background_fill_dark="#fdfaf4",
    block_border_width="2px",
    block_border_color="#8b7355",
    block_border_color_dark="#8b7355",
    block_radius="0px",
    block_shadow="4px 4px 0px rgba(139, 115, 85, 0.4)",
    button_primary_background_fill="#c4a482",
    button_primary_background_fill_dark="#c4a482",
    button_primary_text_color="#3e3222",
    button_primary_text_color_dark="#3e3222",
    button_primary_border_color="#8b7355",
    button_primary_border_color_dark="#8b7355",
    button_secondary_background_fill="#faeedd",
    button_secondary_background_fill_dark="#faeedd",
    button_secondary_text_color="#3e3222",
    button_secondary_text_color_dark="#3e3222",
    button_secondary_border_color="#8b7355",
    button_secondary_border_color_dark="#8b7355",
)

with gr.Blocks(title="Handwritten Digit Recognition") as demo:
    gr.Markdown("# ✍️ Character Recognition AI")
    gr.Markdown("<p class='subtitle'>Draw a digit (0-9) below. The Deep Learning CNN will predict it and generate a <b>Grad-CAM Heatmap</b> to show its exact neural activations!</p>")
    
    with gr.Row():
        with gr.Column(scale=1):
            sketchpad = gr.Sketchpad(type="numpy", label="Draw here!", brush=gr.Brush(colors=["#000000"], default_size=15))
            predict_btn = gr.Button("Predict Number", variant="primary", size="lg")
            
        with gr.Column(scale=1):
            output_text = gr.Markdown("Predicted Number: -- Confidence: --")
            output_image = gr.Image(label="Grad-CAM Heatmap (What the model 'sees')")
            
    predict_btn.click(
        fn=predict_and_explain,
        inputs=sketchpad,
        outputs=[output_text, output_image]
    )

if __name__ == "__main__":
    if model is None:
        print("Warning: Model not found. The app will launch but cannot make predictions until you run train_model.py")
    print("Launching Interactive Drawing App...")
    demo.launch(theme=vintage_theme, css=custom_css)
