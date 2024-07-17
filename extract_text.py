import fitz  # PyMuPDF
from PIL import Image
import io
import cv2
import matplotlib.pyplot as plt
import numpy as np
from paddleocr import PaddleOCR, draw_ocr

# Specify paths to the local models
detection_model_dir = 'D:\\Users\\Ge2\\source\\repos\\paddleocr\\model\\en_PP-OCRv3_det_slim_distill_train'
recognition_model_dir = 'D:\\Users\\Ge2\\source\\repos\\paddleocr\\model\\en_PP-OCRv3_rec_slim_train'

# Initialize PaddleOCR with local models
ocr = PaddleOCR(
    det_model_dir=detection_model_dir,
    rec_model_dir=recognition_model_dir,
    use_angle_cls=True,
    lang='en'
)

# Function to extract images from PDF
def extract_images_from_pdf(pdf_path):
    document = fitz.open(pdf_path)
    images = []
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        for img in page.get_images(full=True):
            xref = img[0]
            base_image = document.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(io.BytesIO(image_bytes))
            images.append(image)
    return images

# Function to perform OCR on extracted images
def ocr_on_images(images):
    results = []
    for i, image in enumerate(images):
        image_array = np.array(image)
        result = ocr.ocr(image_array, cls=True)
        results.append(result)
        # Print the results
        for line in result:
            print(line)
        # Optionally, visualize the results
        boxes = [element[0] for element in result[0]]
        txts = [element[1][0] for element in result[0]]
        scores = [element[1][1] for element in result[0]]
        im_show = draw_ocr(image_array, boxes, txts, scores, font_path='C:\\Windows\\Fonts\\Arial.ttf')
        im_show = cv2.cvtColor(im_show, cv2.COLOR_RGB2BGR)
        plt.imshow(im_show)
        plt.title(f"Image {i+1}")
        plt.axis('off')
        plt.show()
    return results

# Main execution
if __name__ == "__main__":
    pdf_path = "Testing.pdf"  # Path to your PDF file
    images = extract_images_from_pdf(pdf_path)
    ocr_results = ocr_on_images(images)
