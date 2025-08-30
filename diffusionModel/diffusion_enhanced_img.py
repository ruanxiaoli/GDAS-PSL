# import cv2
# import numpy as np
# import os
#
## Define the directories
# base_dir = 'ndata/xiasina/hpa/enhanced/'
# dir1 = os.path.join(base_dir, 'inensity_level_weak_restore')
# dir2 = os.path.join(base_dir, 'inensity_level_weak')
# output_dir = os.path.join(base_dir, 'inensity_level_weak_add')
#
##
# dir1 = os.path.join(base_dir, 'inensity_level_moderate_restore')
##dir2 = os.path.join(base_dir, 'inensity_level_moderate')
##output_dir = os.path.join(base_dir, 'inensity_level_moderate_add')
##
##dir1 = os.path.join(base_dir, 'inensity_level_strong_restore')
##dir2 = os.path.join(base_dir, 'inensity_level_strong')
##output_dir = os.path.join(base_dir, 'inensity_level_strong_add')
#
## Get the list of subdirectories in each directory
# subdirs1 = [d for d in os.listdir(dir1) if os.path.isdir(os.path.join(dir1, d))]
# subdirs2 = [d for d in os.listdir(dir2) if os.path.isdir(os.path.join(dir2, d))]
#
## Process each pair of subdirectories
# for subdir1, subdir2 in zip(subdirs1, subdirs2):
#    # Construct the full subdirectory paths
#    path1 = os.path.join(dir1, subdir1)
#    path2 = os.path.join(dir2, subdir2)
#    output_path = os.path.join(output_dir, subdir1)  # Use the same subdirectory name
#
#    # Create the output subdirectory if it doesn't exist
#    os.makedirs(output_path, exist_ok=True)
#
#    # Get the list of files in each subdirectory
#    files1 = os.listdir(path1)
#    files2 = os.listdir(path2)
#
#    # Process each pair of files
#    for file1, file2 in zip(files1, files2):
#        # Construct the full file paths
#        file_path1 = os.path.join(path1, file1)  # First image path
#        file_path2 = os.path.join(path2, file2)  # Second image path
#        output_file_path = os.path.join(output_path, file1)  # Output image path
#
#        # Read the images
#        image1 = cv2.imread(file_path1)
#        image2 = cv2.imread(file_path2)
#
#        # Check if the images were successfully read
#        if image1 is None or image2 is None:
#            print(f"Unable to read images: {file_path1}, {file_path2}")
#            continue
#
#        # Resize the images to be the same size if they are not
#        if image1.shape != image2.shape:
#            image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
#
#        # Convert the images to float32 in order to perform alpha blending
#        image1 = np.float32(image1)
#        image2 = np.float32(image2)
#
#        # Define the alpha and beta parameters for alpha blending
##        High
##        alpha = 0.00001
##        #        Medium
##        alpha = 0.0001
##        #        Low
#        alpha = 0.0001
#        beta = 1.0 - alpha
#
#        # Perform alpha blending
#        result_image = cv2.addWeighted(image1, alpha, image2, beta, 0.0)
#
#        # Convert the result image back to uint8
#        result_image = np.uint8(result_image)
#
#        # Save and display the result image
#        cv2.imwrite(output_file_path, result_image)
##
##        cv2.imshow('Result Image', result_image)
##        cv2.waitKey(0)
##        cv2.destroyAllWindows()
import cv2
import numpy as np
import os
import sys
import csv


def main():
    # Define base directory
    count = 0
    base_dir = 'ndata/xiasina/hpa/enhanced/'
    file_name = []

    # Define specific intensity level directories
    # Uncomment the corresponding intensity level as needed and comment out others
    # Weak intensity level
    dir1 = os.path.join(base_dir, 'inensity_level_weak_restore')  # Files containing '_X'
    dir2 = os.path.join(base_dir, 'inensity_level_weak')  # Files not containing '_X'
    output_dir = os.path.join(base_dir, 'inensity_level_weak_add_0.001')  # Output directory

    # Medium intensity level
    # dir1 = os.path.join(base_dir, 'inensity_level_moderate_restore')
    # dir2 = os.path.join(base_dir, 'inensity_level_moderate')
    # output_dir = os.path.join(base_dir, 'inensity_level_moderate_add')

    # Strong intensity level
    # dir1 = os.path.join(base_dir, 'inensity_level_strong_restore')
    # dir2 = os.path.join(base_dir, 'inensity_level_strong')
    # output_dir = os.path.join(base_dir, 'inensity_level_strong_add')

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get subdirectory lists from dir1 and dir2
    subdirs1 = [d for d in os.listdir(dir1) if os.path.isdir(os.path.join(dir1, d))]
    subdirs2 = [d for d in os.listdir(dir2) if os.path.isdir(os.path.join(dir2, d))]

    # Check if subdirectory names in dir1 and dir2 match
    subdirs1_set = set(subdirs1)
    subdirs2_set = set(subdirs2)
    common_subdirs = subdirs1_set.intersection(subdirs2_set)

    if not common_subdirs:
        print("Warning: No matching subdirectories found. Please check if subdirectory names in dir1 and dir2 are consistent.")
        sys.exit(1)
    else:
        if len(common_subdirs) < len(subdirs1_set) or len(common_subdirs) < len(subdirs2_set):
            print("Warning: Some subdirectories do not have corresponding matches in dir1 and dir2. Only processing matching subdirectories.")

    # Process each matching subdirectory
    for subdir in common_subdirs:

        path1 = os.path.join(dir1, subdir)  # inensity_level_weak_restore/subdirectory
        path2 = os.path.join(dir2, subdir)  # inensity_level_weak/subdirectory
        output_subdir = os.path.join(output_dir, subdir)  # inensity_level_weak_add/subdirectory

        # Create output subdirectory
        os.makedirs(output_subdir, exist_ok=True)

        # Get file list from dir2 subdirectory and build mapping based on base names
        files2 = [f for f in os.listdir(path2) if os.path.isfile(os.path.join(path2, f))]
        files2_mapping = {}
        for file2 in files2:
            basename2, ext2 = os.path.splitext(file2)
            files2_mapping[basename2] = file2  # Base name mapping to file name

        # Get file list from dir1 subdirectory and build mapping based on base names
        files1 = [f for f in os.listdir(path1) if os.path.isfile(os.path.join(path1, f))]
        files1_mapping = {}
        for file1 in files1:
            basename1, ext1 = os.path.splitext(file1)
            parts = basename1.rsplit('_', 1)
            if len(parts) == 2:
                prefix, begin_str = parts
                try:
                    begin = float(begin_str)
                    if prefix in files1_mapping:
                        files1_mapping[prefix].append((file1, begin))
                    else:
                        files1_mapping[prefix] = [(file1, begin)]
                except ValueError:
                    # If cannot convert to float, treat as not having '_X'
                    prefix = basename1
                    begin = None
                    if prefix in files1_mapping:
                        files1_mapping[prefix].append((file1, begin))
                    else:
                        files1_mapping[prefix] = [(file1, begin)]
            else:
                # File name does not contain '_X'
                prefix = basename1
                begin = None
                if prefix in files1_mapping:
                    files1_mapping[prefix].append((file1, begin))
                else:
                    files1_mapping[prefix] = [(file1, begin)]

        # Iterate through all files in dir2
        for basename2, file2 in files2_mapping.items():
            file_path2 = os.path.join(path2, file2)

            # Check if dir1 has corresponding files with '_X'
            if basename2 in files1_mapping:
                corresponding_files = files1_mapping[basename2]
                # Filter files with '_X'
                files_with_X = [(file1, begin) for file1, begin in corresponding_files if begin is not None]

                if files_with_X:
                    # If files with '_X' exist, prioritize processing these files
                    # Sort by 'X' value, prioritize smaller 'X' values (e.g., -1, 0, 1, ...)
                    files_with_X_sorted = sorted(files_with_X, key=lambda x: x[1])
                    # Assume prioritizing the first qualifying file
                    file1, begin = files_with_X_sorted[0]

                    if begin in [-1, 0]:
                        # If 'begin' is -1 or 0, directly save dir2's image to output directory
                        image2 = cv2.imread(file_path2)
                        if image2 is None:
                            print(f"Error: Unable to read image: {file_path2}")
                            sys.exit(1)
                        new_file_name = basename2 + os.path.splitext(file2)[1]  # Remove '_X'
                        output_file_path = os.path.join(output_subdir, new_file_name)
                        cv2.imwrite(output_file_path, image2)
                        print(f"Saved non-blended image: {output_file_path}")
                    else:
                        # Otherwise, perform alpha blending
                        count += 1
                        file_path1_specific = os.path.join(path1, file1)
                        file_name.append(file_path1_specific)
                        image1 = cv2.imread(file_path1_specific)
                        image2 = cv2.imread(file_path2)

                        if image1 is None:
                            print(f"Error: Unable to read image: {file_path1_specific}")
                            sys.exit(1)
                        if image2 is None:
                            print(f"Error: Unable to read image: {file_path2}")
                            sys.exit(1)

                        # If image dimensions are inconsistent, resize image2 to match image1
                        if image1.shape != image2.shape:
                            image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
                            print(f"Resized image: {file_path2} to match {file_path1_specific}")

                        # Convert images to float32 for precise blending
                        image1_float = np.float32(image1)
                        image2_float = np.float32(image2)

                        # Define alpha and beta parameters for alpha blending
                        alpha_blend = 0.001  # Weight for image1
                        beta_blend = 1.0 - alpha_blend  # Weight for image2

                        # Perform alpha blending
                        result_image = cv2.addWeighted(image1_float, alpha_blend, image2_float, beta_blend, 0.0)

                        # Convert result image back to uint8
                        result_image_uint8 = np.uint8(result_image)

                        # Create new file name, removing '_X' part
                        new_file_name = basename2 + os.path.splitext(file2)[1]
                        output_file_path = os.path.join(output_subdir, new_file_name)

                        # Save blended image to output directory
                        cv2.imwrite(output_file_path, result_image_uint8)
                        print(f"Saved blended image: {output_file_path}")
                else:
                    # If no files with '_X' found, report error and terminate program
                    print(f"Error: No corresponding files with '_X' found in dir1 for {basename2}")
                    sys.exit(1)
            else:
                # If no corresponding file found in dir1, report error and terminate program
                print(f"Error: No corresponding file found in dir1 for {basename2}")
                sys.exit(1)
    return count, file_name


if __name__ == "__main__":
    count, file_names = main()
    csv_file_path = 'file_names.csv'
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['File Name'])  # Write header, can be omitted if only storing file names
        for file_name in file_names:
            writer.writerow([file_name])  # Write each file name as a row

    print(f"File names have been saved as CSV file: {csv_file_path}")
    print(count)