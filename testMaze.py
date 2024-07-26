'''
python -m venv myenv

source myenv/bin/activate

pip install -r requirements.txt

python testMaze.py fig_Maze1.png
python testMaze.py fig_Maze2.png

fonte: https://journals.aps.org/prx/pdf/10.1103/PhysRevX.14.031005
python testMaze.py fig_Maze3a.png
python testMaze.py fig_Maze3b.png

'''
import time
import argparse
from morph import *


def main(input_file):
    # Read and preprocess the image
    img0 = mm.read(input_file)

    if input_file == 'fig_Maze1.png':
        img0 = cv2.resize(img0, (35, 35))
    elif input_file == 'fig_Maze2.png':
        img0 = mm.ero(img0, mm.sebox())
        img0 = cv2.resize(img0, (700, 700))

    imgRGB = mm.color(img0)
    img = mm.gray(imgRGB)
    f = (mm.threshold(img) / 255).astype('uint16')

    if input_file == 'fig_Maze2.png':
        f = mm.open(f, mm.secross())
        f1 = (mm.neg(f) / 255).astype('uint8')
        f2 = mm.label(f1)

    # Display the preprocessed image
    plt.figure(figsize=(10, 10))
    mm.show(f)  # image (a)
    print(f.shape)
    H,W = f.shape

    # Define markers based on the input file
    m1, m2 = np.zeros_like(f, dtype='uint8'), np.zeros_like(f, dtype='uint8')

    if input_file == 'fig_Maze1.png':
        m1[1, 17], m2[-2, 17] = 1, 1
        mm.show(imgRGB, mm.dil(m1, mm.secross()), mm.dil(m2, mm.secross()))
    elif input_file == 'fig_Maze2.png':
        m1, m2 = np.array(f2 == 2, dtype='uint8'), np.array(f2 == 3, dtype='uint8')
        mm.show(imgRGB, mm.dil(m1, mm.sedisk(2)), mm.dil(m2, mm.sedisk(2)))
    elif input_file == 'fig_Maze3a.png':
        m1[2030, 2065], m2[3400, 1250] = 1, 1
        mm.show(imgRGB, mm.dil(m1, mm.sedisk(30)), mm.dil(m2, mm.sedisk(30)))
    elif input_file == 'fig_Maze3b.png':
        m1[301, 297], m2[485, 175] = 1, 1
        mm.show(imgRGB, mm.dil(m1, mm.sedisk(7)), mm.dil(m2, mm.sedisk(7)))

    #exit(0)
    # Measure the processing time
    start_time = time.time()
    if input_file == 'fig_Maze1.png':
        b = mm.gdist(f, m1)  # image (b)
    else:
        b = mm.gdist(f, m1, mm.secross())  # image (b)
    end_time = time.time()
    processing_time = end_time - start_time
    print(f"Processing time for m1: {processing_time:.4f} seconds")

    if input_file == 'fig_Maze1.png':
        c = mm.gdist(f, m2)  # image (c)
    else:
        c = mm.gdist(f, m2, mm.secross())  # image (c)

    # Display the results
    mm.show(b)
    mm.show(c)

    # Calculate and display the minimum distance
    d = b + c
    min_dist = np.amin(d[d != np.amin(d)])
    print(f'Minimum distance: {min_dist}')
    mm.show(imgRGB, d == min_dist)  # image (d)


if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Process an image with geodesic distance transform.')
    parser.add_argument('file', type=str, help='Path to the image file to be processed')
    args = parser.parse_args()

    # Run the main function with the provided file
    main(args.file)
