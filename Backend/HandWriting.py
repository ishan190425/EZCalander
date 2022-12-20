import cv2
import pytesseract
import numpy as np

class HandWriting():
    def __init__(self) -> None:
        pass

    def read_handwritten_table(self,image_path):
        # Load the image and convert it to grayscale
        image = cv2.imread(image_path)
    
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply image enhancement techniques to improve the quality of the image
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        
        # Identify and extract the region of the image that contains the table
        # This can be done using techniques such as contour detection or template matching
        table_region = self.extract_table_region(thresh)
        
        # Identify the lines that define the rows and columns of the table
        # This can be done using techniques such as edge detection or Hough line transformation
        rows, cols = self.identify_table_lines(table_region)
        
        # Divide the table into individual cells based on the identified lines
        cells = self.divide_into_cells(table_region, rows, cols)
        
        data = self.read_data_from_cells(cells)
        
        # Organize the data into a structured format, such as a 2D array or a pandas DataFrame
        data = self.organize_data(data, rows, cols)
        
        return data

    def read_data_from_cells(self, cells):
         # Initialize an empty list to store the data from the table
        data = []
        # Iterate over the cells and extract the text using tesseract
        for i, cell in enumerate(cells):
            try:
                cell_text = pytesseract.image_to_string(cell)
                data.append(cell_text)
            except:
                continue
        return data

    def extract_table_region(self,image):
        # Find the contours in the image
        contours = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        
        # Find the contour with the largest area, which is likely to be the table
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Extract the bounding rectangle of the table
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Crop the image to the bounding rectangle
        table_region = image[y:y+h, x:x+w]
        
        return table_region

    def identify_table_lines(self,image):
        
        # Apply Gaussian blur to smooth the image and reduce noise
        blur = cv2.GaussianBlur(image, (5, 5), 0)
        
        # Apply Canny edge detection to find the edges in the image
        edges = cv2.Canny(blur, 50, 150)
        
        # Find the lines in the image using Hough line transformation
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
        
        # Extract the rows and columns of the table
        rows, cols = self.extract_rows_and_cols(lines)
        
        return rows, cols

    def extract_rows_and_cols(self,lines):
    # Initialize empty lists to store the coordinates of the rows and columns
        rows = []
        cols = []
        
        # Iterate over the lines and categorize them as rows or columns based on their orientation
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(x1 - x2) > abs(y1 - y2):
                cols.append(line)
            else:
                rows.append(line)
        
        # Sort the rows and columns by their coordinates
        rows.sort(key=lambda x: min(x[0][1], x[0][3]))
        cols.sort(key=lambda x: min(x[0][0], x[0][2]))
        
        return rows, cols

    def organize_data(self,data, rows, cols):
        # Initialize an empty 2D array to store the organized data
        organized_data = [[None for _ in range(len(cols))] for _ in range(len(rows))]
        
        # Iterate over the cells and organize the data into the 2D array
        for i, cell in enumerate(data):
            row = i // len(cols)
            col = i % len(cols)
            organized_data[row][col] = cell
    
        return organized_data

    def divide_into_cells(self, image, rows, cols):
        # Convert the rows and columns to a form that is easier to manipulate
        rows = [line[0][1] for line in rows]
        rows.append(image.shape[0])
        cols = [line[0][0] for line in cols]
        cols.append(image.shape[1])
        
        # Initialize a list to store the cells
        cells = []
        
        # Iterate over the rows and columns to extract the individual cells
        for i in range(len(rows) - 1):
            for j in range(len(cols) - 1):
                # Crop the image to the current cell
                cell = image[rows[i]:rows[i+1], cols[j]:cols[j+1]]
                
                # Add the cell to the list
                cells.append(cell)
        
        return cells



if __name__ == "__main__":
    h = HandWriting()
    data = h.read_handwritten_table("/home/rflix/Coding/EZCalander/Examples/example2.jpg")
    print(f"Data: {data}")
