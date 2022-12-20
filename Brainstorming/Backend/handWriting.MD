# Backend
## Classes
    - API 
    - DataType
    - Handwriting Reconzation
    - InputClass (Connect to FrontEnd)

    Pre-processing:

    Convert the input image to grayscale and apply image enhancement techniques such as thresholding or contrast stretching to improve the quality of the image.
    Identify and extract the region of the image that contains the table.

    Table structure analysis:

    Identify the lines that define the rows and columns of the table. This can be done using techniques such as edge detection or Hough line transformation.
    Divide the table into individual cells based on the identified lines.

    Text recognition:

    Identify the text within each cell using techniques such as OCR (optical character recognition).
    You can use a machine learning model trained on a large dataset of handwritten text to improve the accuracy of the OCR process.

    Data formatting:

    Parse the recognized text to extract the data from the table and organize it into a structured format, such as a 2D array or a pandas DataFrame.