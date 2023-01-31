# OCR

## Flow
- Image processing to find contours (GaussianBlur, Canny, dilate, and threshold)
- Find the table from contours by select the widest area
- Crop the table
- Apply OCR by using tesseract
- Collect texts and coordinates from tesseract
- Aplly clustering method to cluster all texts that have similiar x-coordinate and define them as one column
- define confident threshold and distance threshold to get better result
- input all data to table

## Bottleneck
- Difficult to get parameters that can perfectly describe the contours of the table.
- Tesseract can't detect handwriting very well.
- There are some images that are of poor quality
- 
## Area Improvement
- Perform parameter analysis to find a nicer table contour.
- Uses an object detection model to detect table and text.

**I have tried:** Built a YOLO model for text detection, but the results are not very good because it uses little data and takes a long time to annotate images.
Built a handwriting classification model from Kaggle dataset but can't read the image properly unless it has to be cropped for each cell first.
