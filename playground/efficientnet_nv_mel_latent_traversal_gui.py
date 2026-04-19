import sys
import os
import torch
import numpy as np
import gradio as gr
import matplotlib.cm as cm
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

class BinaryTarget:
    def __init__(self, target_class=1):
        self.target_class = target_class
    def __call__(self, model_output):
        if self.target_class == 1:
            return model_output[0]
        else:
            return -model_output[0]

from efficientnet_nv_mel_latent_traversal import (
    ae_model, 
    classifier_model,
    val_orig_dataset, 
    LABEL_NAMES, 
    DEVICE,
    _recon_error,
    _recon_loop_loss,
    take_latent_step,
    take_pca_step
)

def launch_gradio_app():
    print("Preparing gallery images and computing dataset embeddings...")
    mean_t = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to("cpu")
    std_t  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to("cpu")
    
    gallery_items = []
    all_imgs = []
    with torch.no_grad():
        for idx in range(len(val_orig_dataset)):
            item = val_orig_dataset[idx]
            img_t = item["image"]
            lbl_t = item["label"]
            img_disp = (img_t * std_t + mean_t).clamp(0, 1).permute(1, 2, 0).numpy()
            lbl_idx = lbl_t.argmax().item()
            gallery_items.append((img_disp, f"{LABEL_NAMES[lbl_idx]}"))
            
            all_imgs.append(img_t)
            
        all_imgs_batch = torch.stack(all_imgs, dim=0).to(DEVICE)
        all_embeddings = ae_model.encoder(all_imgs_batch)

    with gr.Blocks() as demo:
        gr.Markdown("# Latent Traversal Interactive Interface")
        
        current_z_state = gr.State(None)
        orig_recon_state = gr.State(None)
        
        with gr.Tab("Selection & Traversal"):
            gallery = gr.Gallery(value=gallery_items, label="Validation Dataset (Click to Select & Start)", columns=8, allow_preview=False)
            
            with gr.Row():
                btn_step_mel = gr.Button("Step Towards MEL (Gradient IN)")
                btn_step_nv = gr.Button("Step Towards NV (Gradient AGAINST)")
                
            with gr.Row():
                btn_pca_pos_1 = gr.Button("Step +PCA Base 1")
                btn_pca_neg_1 = gr.Button("Step -PCA Base 1")
                btn_pca_pos_2 = gr.Button("Step +PCA Base 2")
                btn_pca_neg_2 = gr.Button("Step -PCA Base 2")
            
            with gr.Row():
                chk_orthogonal = gr.Checkbox(label="Orthogonal", value=False)
                lr_slider = gr.Slider(0.001, 10.0, value=1.0, label="Learning Rate (Step Size)")
                n_steps_slider = gr.Slider(1, 100, value=1, step=1, label="Number of Steps")
                
            with gr.Row():
                orig_img_display = gr.Image(label="Selected Original Image")
                orig_recon_display = gr.Image(label="Original Reconstruction")
                out_img = gr.Image(label="Current Decoded Image")
                diff_map_display = gr.Image(label="Difference Map (recon vs current)")
                
            out_info = gr.Textbox(label="Current Info")
            
            def get_diff_map(orig_rgb, current_rgb):
                lum_weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
                diff_map = ((current_rgb - orig_rgb) * lum_weights).sum(axis=-1)
                abs_max = max(np.abs(diff_map).max(), 1e-6)
                norm_diff = (diff_map + abs_max) / (2 * abs_max)
                return cm.RdBu_r(norm_diff)[:, :, :3]
            
            def on_gallery_select(evt: gr.SelectData):
                idx = evt.index
                item = val_orig_dataset[idx]
                img_t = item["image"].to(DEVICE)
                lbl_t = item["label"]
                lbl_idx = lbl_t.argmax().item()
                
                with torch.no_grad():
                    z_initial = ae_model.encoder(img_t.unsqueeze(0))
                    x_recon = ae_model.decoder(z_initial)
                
                logit = classifier_model(x_recon).squeeze(-1)
                prob = torch.sigmoid(logit).item()
                err = _recon_error(x_recon)
                
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(DEVICE)
                std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(DEVICE)
                
                orig_disp = (img_t * std + mean).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
                sq_disp = (x_recon[0] * std + mean).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
                
                info_text = f"Class: {LABEL_NAMES[lbl_idx]} | Prob MEL: {prob:.4f} | Recon Err: {err:.4f}"
                
                diff_colored = get_diff_map(sq_disp, sq_disp)
                
                return orig_disp, sq_disp, sq_disp, diff_colored, info_text, z_initial.detach(), sq_disp
                
            gallery.select(on_gallery_select, inputs=[], outputs=[orig_img_display, orig_recon_display, out_img, diff_map_display, out_info, current_z_state, orig_recon_state])
            
            def take_step_gui(z, lr, n_steps, orthogonal, direction_mode, orig_recon_disp):
                if z is None:
                    return None, None, "Please select an image first.", None
                
                z_ = z.detach().clone()
                z_.requires_grad = True
                
                optimizer = torch.optim.SGD([z_], lr=lr)
                
                for _ in range(n_steps):
                    take_latent_step(
                        z_, 
                        classifier_model, 
                        ae_model, 
                        _recon_loop_loss, 
                        direction_mode, 
                        optimizer, 
                        orthogonal
                    )
                    
                new_z = z_.detach()
                
                with torch.no_grad():
                    x_recon = ae_model.decoder(new_z)
                    new_logit = classifier_model(x_recon).squeeze(-1)
                    prob = torch.sigmoid(new_logit).item()
                    err = _recon_error(x_recon)
                    
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(DEVICE)
                std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(DEVICE)
                sq_disp = (x_recon[0] * std + mean).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
                
                diff_colored = get_diff_map(orig_recon_disp, sq_disp)
                
                return sq_disp, diff_colored, f"Prob MEL: {prob:.4f} | Recon Err: {err:.4f}", new_z
                
            def take_pca_step_gui(z, lr, n_steps, base_idx, sign, orig_recon_disp):
                if z is None:
                    return None, None, "Please select an image first.", None
                
                z_ = z.detach().clone()
                
                for _ in range(n_steps):
                    z_ = take_pca_step(
                        z_, 
                        classifier_model, 
                        ae_model, 
                        all_embeddings, 
                        base_idx, 
                        sign, 
                        lr
                    )
                    
                new_z = z_
                
                with torch.no_grad():
                    x_recon = ae_model.decoder(new_z)
                    new_logit = classifier_model(x_recon).squeeze(-1)
                    prob = torch.sigmoid(new_logit).item()
                    err = _recon_error(x_recon)
                    
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(DEVICE)
                std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(DEVICE)
                sq_disp = (x_recon[0] * std + mean).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
                
                diff_colored = get_diff_map(orig_recon_disp, sq_disp)
                
                return sq_disp, diff_colored, f"Prob MEL: {prob:.4f} | Recon Err: {err:.4f}", new_z
                
            btn_step_mel.click(take_step_gui, inputs=[current_z_state, lr_slider, n_steps_slider, chk_orthogonal, gr.State(1.0), orig_recon_state], outputs=[out_img, diff_map_display, out_info, current_z_state])
            btn_step_nv.click(take_step_gui, inputs=[current_z_state, lr_slider, n_steps_slider, chk_orthogonal, gr.State(-1.0), orig_recon_state], outputs=[out_img, diff_map_display, out_info, current_z_state])
            
            btn_pca_pos_1.click(take_pca_step_gui, inputs=[current_z_state, lr_slider, n_steps_slider, gr.State(0), gr.State(1.0), orig_recon_state], outputs=[out_img, diff_map_display, out_info, current_z_state])
            btn_pca_neg_1.click(take_pca_step_gui, inputs=[current_z_state, lr_slider, n_steps_slider, gr.State(0), gr.State(-1.0), orig_recon_state], outputs=[out_img, diff_map_display, out_info, current_z_state])
            btn_pca_pos_2.click(take_pca_step_gui, inputs=[current_z_state, lr_slider, n_steps_slider, gr.State(1), gr.State(1.0), orig_recon_state], outputs=[out_img, diff_map_display, out_info, current_z_state])
            btn_pca_neg_2.click(take_pca_step_gui, inputs=[current_z_state, lr_slider, n_steps_slider, gr.State(1), gr.State(-1.0), orig_recon_state], outputs=[out_img, diff_map_display, out_info, current_z_state])

        with gr.Tab("Interpolation"):
            interp_gallery = gr.Gallery(value=gallery_items, label="Validation Dataset (Click to Select)", columns=8, allow_preview=False)
            
            with gr.Row():
                with gr.Column():
                    interp_selected_display = gr.Image(label="Currently Selected Image")
                    with gr.Row():
                        btn_set_a = gr.Button("Set as Image A")
                        btn_set_b = gr.Button("Set as Image B")
                
                with gr.Column():
                    img_a_display = gr.Image(label="Image A")
                    img_b_display = gr.Image(label="Image B")
            
            with gr.Row():
                n_interp_slider = gr.Slider(1, 20, value=5, step=1, label="Number of intermediate images")
                btn_interp = gr.Button("Interpolate")
                
            interp_results = gr.Gallery(label="Interpolation Results", columns=7, allow_preview=True)
            interp_gradcam_results = gr.Gallery(label="Grad-CAM Overlay (MEL)", columns=7, allow_preview=True)
            
            interp_state_selected = gr.State(None)
            interp_state_a = gr.State(None)
            interp_state_b = gr.State(None)
            
            def on_interp_gallery_select(evt: gr.SelectData):
                idx = evt.index
                img_disp = gallery_items[idx][0]
                return img_disp, idx
                
            interp_gallery.select(on_interp_gallery_select, inputs=[], outputs=[interp_selected_display, interp_state_selected])
            
            def set_image_a(idx):
                if idx is None:
                    return None, None
                return gallery_items[idx][0], idx
                
            def set_image_b(idx):
                if idx is None:
                    return None, None
                return gallery_items[idx][0], idx
                
            btn_set_a.click(set_image_a, inputs=[interp_state_selected], outputs=[img_a_display, interp_state_a])
            btn_set_b.click(set_image_b, inputs=[interp_state_selected], outputs=[img_b_display, interp_state_b])
            
            def do_interpolation(idx_a, idx_b, n_steps):
                if idx_a is None or idx_b is None:
                    return [], []
                
                img_a_t = val_orig_dataset[idx_a]["image"].to(DEVICE)
                img_b_t = val_orig_dataset[idx_b]["image"].to(DEVICE)
                
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(DEVICE)
                std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(DEVICE)
                
                with torch.no_grad():
                    z_a = ae_model.encoder(img_a_t.unsqueeze(0))
                    z_b = ae_model.encoder(img_b_t.unsqueeze(0))
                    alphas = torch.linspace(0, 1, int(n_steps) + 2).to(DEVICE)
                    
                results = []
                gradcam_results = []
                
                target_layers = [classifier_model.backbone.features[-2]]
                cam = GradCAM(model=classifier_model, target_layers=target_layers)
                targets = [BinaryTarget(target_class=1)]
                
                for i, alpha in enumerate(alphas):
                    with torch.no_grad():
                        z_interp = (1.0 - alpha) * z_a + alpha * z_b
                        x_recon = ae_model.decoder(z_interp)
                        logit = classifier_model(x_recon).squeeze(-1)
                        prob = torch.sigmoid(logit).item()
                        
                        sq_disp = (x_recon[0] * std + mean).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
                        
                    caption = f"Prob MEL: {prob:.4f}"
                    if i == 0:
                        caption = "Recon A | " + caption
                    elif i == len(alphas) - 1:
                        caption = "Recon B | " + caption
                    else:
                        caption = f"Step {i} | " + caption
                        
                    results.append((sq_disp, caption))
                    
                    cam_input = x_recon.detach().clone()
                    grayscale_cam = cam(input_tensor=cam_input, targets=targets)
                    grayscale_cam = grayscale_cam[0, :]
                    visualization = show_cam_on_image(sq_disp, grayscale_cam, use_rgb=True)
                    gradcam_results.append((visualization, caption))
                        
                return results, gradcam_results

            btn_interp.click(do_interpolation, inputs=[interp_state_a, interp_state_b, n_interp_slider], outputs=[interp_results, interp_gradcam_results])

    demo.launch(server_name="0.0.0.0", share=False)

if __name__ == "__main__":
    launch_gradio_app()
