from src.plate_detection import PlateDetector
import cv2
import os

def main():
    # Initialize the plate detector
    detector = PlateDetector()

    # Load the test image
    image_path = os.path.join('data', 'sample_images','test.jpg')
    image = cv2.imread(image_path)

    if image is None: 
        print(f"Image not found at {image_path}")
        return
    
    # Detect plates
    detected_plates = detector.detect_plates(image)

    print(f"Detected {len(detected_plates)} plate(s)")

    # cleanup 
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main() 
    