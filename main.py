
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import pytesseract

# Set Tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

app = FastAPI()

# Database of authorized plates
ALLOWED_PLATES = ["ABC1234", "CAR2025", "HR26FC2782", "MH12DE1433","KL6036","NWKL6036","QL9904","PFQ5217"]

@app.post("/upload")
async def upload(request: Request):
    image_bytes = await request.body()
    np_arr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if image is None:
        return JSONResponse(status_code=400, content={"error": "Invalid image"})

    # Show live feed window
    cv2.imshow("ESP32-CAM Stream", image)
    cv2.waitKey(30) 

    # Image Processing for OCR
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(blur, 30, 200)

    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    detected_text = "NOT_FOUND"
    for c in contours:
        approx = cv2.approxPolyDP(c, 10, True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(c)
            cropped = gray[y:y+h, x:x+w]
            # Resize and Threshold for better OCR accuracy
            cropped = cv2.resize(cropped, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            _, cropped = cv2.threshold(cropped, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            detected_text = pytesseract.image_to_string(cropped, config='--psm 8')
            detected_text = ''.join(e for e in detected_text if e.isalnum()).upper()
            break

    print(f"🚗 Detected: {detected_text}")

    status = "ALLOWED" if detected_text in ALLOWED_PLATES else "DENIED"
    return JSONResponse(content={"plate": detected_text, "status": status})
