import cv2
import pytesseract
import numpy as np
import matplotlib.pyplot as plt

class HandWriting():
    def __init__(self) -> None:
        self.index = 1
        self.white = (255,255,255)
        self.blue = (0,0,255)
        self.red = (255,0,0)
        self.green = (0,255,0)
        self.black = (0,0,0)
    
    def show_image(self,image,title):
        plt.imshow(image)
        plt.title(title)
        self.index += 1
        plt.show()
        
        
    def read_handwritten_table(self,image_path):
        # Load the image and convert it to grayscale
        image = cv2.imread(image_path)
        #self.show_image(image,"Original")
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #self.show_image(gray,"Gray - 1")
        
        # Apply image enhancement techniques to improve the quality of the image
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        #self.show_image(thresh,"Threshold - 2")
        
        # Identify and extract the region of the image that contains the table
        # This can be done using techniques such as contour detection or template matching
        table_region = self.extract_table_region(thresh)
        #self.show_image(table_region,"Table Region - 3")
        
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
        
        rho = 1  # distance resolution in pixels of the Hough grid
        
        theta = np.pi / 180  # angular resolution in radians of the Hough grid
        
        threshold = 15  # minimum number of votes (intersections in Hough grid cell)
        
        min_line_length = 80  # minimum number of pixels making up a line
        
        max_line_gap = 3  # maximum gap in pixels between connectable line segments
        
        line_image = np.copy(image) * 0  # creating a blank to draw lines on

        # Run Hough on edge detected image
        # Find the lines in the image using Hough line transformation
        lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)
        
        line_image = np.copy(image)
        
        # Extract the rows and columns of the table
        rows, cols = self.extract_rows_and_cols(lines,line_image,image)
        
        lines_edges = cv2.addWeighted(image, 0.8, line_image, 1, 0)
        
        self.show_image(lines_edges,"Lines")
        
        return rows, cols

    def extract_rows_and_cols(self,lines,line_image,image=None):
    # Initialize empty lists to store the coordinates of the rows and columns
        rows = []
        cols = []
        
        # Iterate over the lines and categorize them as rows or columns based on their orientation
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(x1 - x2) > abs(y1 - y2):
                x2 += 1000
                line[0][2] = x2
                rows.append(line)
                print("Row")
                #x1 -= 1000
            else:
                y2 += 1000
                y1 -= 1000
                line[0][1] = y1
                line[0][3] = y2
                cols.append(line)
                print("Col")
            cv2.line(line_image, (x1, y1), (x2, y2), self.black, 5)
        
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
        rows_threshold = 50 # what is minimum amount of data to be useful - rows
        cols_threshold = 50 # what is minimum amount of data to be useful - cols
        # Initialize a list to store the cells
        cells = []
        
        # Iterate over the rows and columns to extract the individual cells
        for i in range(len(rows) - 1):
            for j in range(len(cols) - 1):
                # Crop the image to the current cell
                rows_total = abs(rows[i] - rows[i+1])
                cols_total = abs(cols[j] - cols[j+1])
                # if rows_total < rows_threshold or cols_total < cols_threshold: #check if cell contains useful data
                #     continue
                cell = image[rows[i]:rows[i+1], cols[j]:cols[j+1]]
                # Add the cell to the list
                cells.append(cell)
                self.show_image(cell,"Cell")
        
        return cells



if __name__ == "__main__":
    h = HandWriting()
    data = h.read_handwritten_table("Examples/example2.jpg")
    print(f"Data: {data}")
