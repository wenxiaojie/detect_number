# Pythono3 code to rename multiple  
# files in a directory or folder 

# importing os module 
import os
import glob

img_list = glob.glob("data/custom/test/*.png")

# Function to rename multiple files 
def main():
    
    for index in range(len(img_list)):
        dst = str.zfill(str(index + 1), 5) + ".png"
        src = 'data/custom/test/' + str(index + 1) + '.png'
        dst = 'data/custom/test_image1/' + dst

        # rename() function will 
        # rename all the files 
        os.rename(src, dst)

    # Driver Code 

if __name__ == '__main__':
    # Calling main() function 
    main() 
