import numpy as np
import cv2
import os
import nibabel as nib

# Lazy MONAI Model initialization
def get_monai_unet():
    from monai.networks.nets import UNet
    model = UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    )
    return model

def load_medical_image(path):
    """
    Loads a medical image (NIfTI, DICOM, or standard image).
    Returns a 2D grayscale numpy array (0-255).
    """
    if path.endswith(('.nii', '.nii.gz')):
        # Load NIfTI volume
        try:
            img_nifti = nib.load(path)
            data = img_nifti.get_fdata()
            
            # Extract middle slice (handle 3D or 4D)
            if len(data.shape) == 3:
                mid = data.shape[2] // 2
                slice_data = data[:, :, mid]
            elif len(data.shape) == 4:
                mid_z = data.shape[2] // 2
                mid_t = data.shape[3] // 2
                slice_data = data[:, :, mid_z, mid_t]
            else:
                slice_data = data
                
            # Normalize to 0-255
            if np.max(slice_data) != np.min(slice_data):
                slice_data = (slice_data - np.min(slice_data)) / (np.max(slice_data) - np.min(slice_data)) * 255.0
            
            # Ensure it's a contiguous uint8 array for OpenCV
            return np.ascontiguousarray(slice_data.astype(np.uint8))
        except Exception as e:
            print(f"Error loading NIfTI: {e}")
            return np.zeros((256, 256), dtype=np.uint8)
    else:
        # Standard image or DICOM placeholder
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            # Fallback for failed loads
            return np.zeros((256, 256), dtype=np.uint8)
        return np.ascontiguousarray(img)

def run_monai_segmentation(input_path, output_path):
    # Load image (handle NIfTI or standard)
    img = load_medical_image(input_path)
    
    # Visual simulation of segmentation (Circle in the center)
    mask = np.zeros_like(img)
    h, w = img.shape
    cv2.circle(mask, (w//2, h//2), min(h, w)//4, (255), -1)
    
    # Overlay for demonstration
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    mask_rgb = np.zeros_like(img_rgb)
    mask_rgb[:,:,1] = mask # Green mask
    
    segmented = cv2.addWeighted(img_rgb, 0.7, mask_rgb, 0.3, 0)
    cv2.imwrite(output_path, segmented)
    return output_path

def run_cyclegan_translation(input_path, output_path, mode="mri2ct"):
    # Load image (handle NIfTI or standard)
    img = load_medical_image(input_path)
    
    if mode == "mri2ct":
        # Simulating CT bone density (Equalization + Bone Colormap)
        translated = cv2.equalizeHist(img)
        translated = cv2.applyColorMap(translated, cv2.COLORMAP_BONE)
    else:
        # Simulating MRI soft tissue (Blur + RGB conversion)
        translated = cv2.GaussianBlur(img, (5,5), 0)
        translated = cv2.cvtColor(translated, cv2.COLOR_GRAY2RGB)
        
    cv2.imwrite(output_path, translated)
    return output_path

def generate_diagnostic_report(modality, atlas_type):
    from groq import Groq
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return "GROQ API Key missing. Please check your .env file."
        
    client = Groq(api_key=api_key)
    prompt = f"As a professional radiologist, provide a concise diagnostic summary for a {modality} scan registered with {atlas_type} atlas. Mention potential structural deformations."
    
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant",
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Error generating report: {str(e)}"
