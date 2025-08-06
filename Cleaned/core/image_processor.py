"""
Image preprocessing utilities for object detection and enhancement
"""
import cv2
import numpy as np


class ImageProcessor:
    """Handles image preprocessing for better detection results"""
    
    @staticmethod
    def enhance_frame(image):
        """Enhance frame with contrast and sharpening"""
        alpha = 1.3
        beta = 30
        image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        kernel = np.array([[0, -1, 0],
                          [-1, 5, -1],
                          [0, -1, 0]])
        return cv2.filter2D(src=image, ddepth=-1, kernel=kernel)
    
    @staticmethod
    def preprocess_for_object_detection(image):
        """
        Preprocess image for object detection.
        Currently returns original image but can be extended for enhancement.
        """
        # Future enhancements can be added here:
        # - Contrast adjustment
        # - Noise reduction
        # - Edge enhancement
        return image
    
    @staticmethod
    def enhance_for_detection(image):
        """
        Enhanced preprocessing for better object detection
        """
        # Convert to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image.copy()
        
        # Increase contrast and brightness
        alpha = 1.4  # Contrast control
        beta = 15    # Brightness control
        contrast_bright = cv2.convertScaleAbs(image_rgb, alpha=alpha, beta=beta)
        
        # Reduce noise while preserving edges
        denoised = cv2.fastNlMeansDenoisingColored(contrast_bright, None, 10, 10, 7, 21)
        
        # Enhance edges
        kernel_sharpen = np.array([[-1,-1,-1], 
                                  [-1, 9,-1],
                                  [-1,-1,-1]])
        sharpened = cv2.filter2D(denoised, -1, kernel_sharpen)
        
        # Convert back to BGR for OpenCV operations
        if len(image.shape) == 3 and image.shape[2] == 3:
            return cv2.cvtColor(sharpened, cv2.COLOR_RGB2BGR)
        return sharpened
