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
    
    def find_plate_candidates(self, edges: np.ndarray, original_image: np.ndarray) -> List[np.ndarray]:
        # Find contours in the image
        contours, _ = cv2.findContours(edges,
                                        cv2.RETR_EXTERNAL, 
                                        cv2.CHAIN_APPROX_SIMPLE)  
        
        plate_candidates = []

        for contour in contours:
            area = cv2.contourArea(contour)

            # Filter out small and large contours
            if area < self.min_plate_area or area > self.max_plate_area:
                continue

            # Get minimum area rectangle 
            rect = cv2.minAreaRect(contour)
            (x, y), (w, h), angle = rect

            # Aspect ratio filtering
            aspect_ratio = max(w, h) / min(w, h)
            if aspect_ratio < self.min_aspect_ratio or aspect_ratio > self.max_aspect_ratio:
                continue

    def extract_plate_regions(self, contours: List[np.ndarray], image: np.ndarray) -> List[np.ndarray]:
        """
        Extract the plate regions from the image based on contours.
        Args:
            image: Original image
            contours: List of plate candidate contours
        Returns:
            List of extracted plate images
        """
        
        plate_regions = []

        for contour in contours:
            # Get minimum area rectangle 
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            # Get width and height of the rectangle
            width = int(rect[1][0])
            height = int(rect[1][1])

            # Handle the case where width and height might be swapped
            if width < height:
                width, height = height, width

            # Source points for perspective transformation
            src_pts = box.astype("float32")

            # Destination points for a straight rectangle
            dest_pts = np.array([[0, height - 1],
                                  [0, 0],
                                  [width - 1, 0],
                                  [width - 1, height - 1]], dtype="float32")
            
            # Calculate perspective transform matrix
            M = cv2.getPerspectiveTransform(src_pts, dest_pts)

            # Warp the image (gives a straight plate image)
            warped = cv2.warpPerspective(image, M, (width, height))

            plate_regions.append(warped)

            if self.debug_mode:
                cv2.imshow('Warped', warped)
                cv2.waitKey(0)

        return plate_regions
    
    def detect_plates(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Detect license plates in the input image.
        Args:
            image: Input BGR image
        Returns:
            List of extracted plate images
        """
        # --- clear pipeline of operations ---
        # Preprocess the image
        edges = self.preprocess_image(image)

        # Find potential plate regions
        plate_candidates = self.find_plate_candidates(edges, image)

        # Extract and return plate regions
        plate_regions = self.extract_plate_regions(plate_candidates, image)

        return plate_regions