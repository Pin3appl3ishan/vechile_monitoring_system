import cv2
import numpy as np
from typing import List, Tuple
import os

class PlateDetector:
    def __init__(self, debug_mode: bool = True):
        # Initialize parameters for plate detection
        self.debug_mode = debug_mode

        self.min_plate_area = 1000
        self.max_plate_area = 15000
        self.min_aspect_ratio = 2.0  # License plates are typically rectangular
        self.max_aspect_ratio = 5.0

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess the image for plate detection.
        Args:
            image: Input BGR image
        Returns:
            Preprocessed binary image
        """
        
        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply bilateral filter to reduce noise while keeping edges sharp
        denoised = cv2.bilateralFilter(gray, 11, 17, 17)
        
        # Edge detection using Canny
        edges = cv2.Canny(denoised, 30, 200)
        
        # Morphological operations to connect nearby edges
        kernel = np.ones((3,3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        if self.debug_mode:
            cv2.imshow('Gray', gray)
            cv2.imshow('Denoised', denoised)
            cv2.imshow('Edges', edges)
            cv2.waitKey(0)
            
        return edges