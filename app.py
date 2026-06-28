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
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def predict_and_explain(img_dict):
    if model is None:
        return "Model not found. Run 'train_model.py' first.", None
        
    if isinstance(img_dict, dict):
        img = img_dict.get('composite', img_dict.get('image', img_dict.get('background')))
    else:
        img = img_dict
        
    if img is None:
        return "Please draw a number.", None
        
    if len(img.shape) == 3:
        img_gray = cv2.cvtColor(img[:,:,:3], cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img
        
    if np.mean(img_gray) > 127:
        img_gray = 255 - img_gray
        
    coords = cv2.findNonZero(img_gray)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        pad = max(10, int(max(w, h) * 0.2))
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(img_gray.shape[1], x + w + pad)
        y2 = min(img_gray.shape[0], y + h + pad)
        img_gray = img_gray[y1:y2, x1:x2]
        
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
    
    return result_text, superimposed_img

custom_css = """
.gradio-container { max-width: 900px !important; }
h1 { text-align: center; background: -webkit-linear-gradient(45deg, #00C9FF, #92FE9D); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 3em !important; margin-bottom: 0 !important; }
.subtitle { text-align: center; font-size: 1.2em; color: #8b949e; margin-bottom: 30px; }
"""
theme = gr.themes.Monochrome(primary_hue="emerald", neutral_hue="slate")

with gr.Blocks(title="Handwritten Digit Recognition") as demo:
    gr.Markdown("# ✍️ Character Recognition AI")
    gr.Markdown("<p class='subtitle'>Draw a digit (0-9) below. The Deep Learning CNN will predict it and generate a <b>Grad-CAM Heatmap</b> to show its exact neural activations!</p>")
    
    with gr.Row():
        with gr.Column():
            sketchpad = gr.Sketchpad(label="Draw here!", type="numpy")
            predict_btn = gr.Button("Predict Number", variant="primary")
            
        with gr.Column():
            output_text = gr.Markdown(label="Prediction Result")
            output_heatmap = gr.Image(label="Grad-CAM Heatmap (What the model 'sees')")
            
    predict_btn.click(predict_and_explain, inputs=sketchpad, outputs=[output_text, output_heatmap])
    
if __name__ == "__main__":
    if model is None:
        print("Warning: Model not found. The app will launch but cannot make predictions until you run train_model.py")
    print("Launching Interactive Drawing App...")
    demo.launch(theme=theme, css=custom_css)
