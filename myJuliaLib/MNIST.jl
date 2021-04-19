using Images
using JSON

"""
    MNIST2json()


IN A NUTSHELL: This function reads MNIST training and test
data from binary files (downloaded from Yann's website)
and then saves them in a .json format. 

DETAILED DESCRIPTION: The training database contains 60000
images, each of size 28*28 pixels. `features` is a 2D
array of size 784 X 60000 whose every column contains
grayscale pixel values of single image, and `labels` is
a column vector of size 60000 whose entries are the
numerical values depicted in each image. The test database
is similarly structured and contains information on 10000
images.

SOURCE: http://yann.lecun.com/exdb/mnist/

INPUT FILES: 
Training data: train-images-idx3-ubyte, train-labels-idx1-ubyte
Test data: t10k-images-idx3-ubyte, 

OUTPUT FILES: MNIST_train.json, MNIST_test.json

APPROXIMATE EXECUTION TIME:
On a MacBook Air (2017) with 1.8 GHz Dual-Core Intel Core i5 
and 8 GB 1600 MHz DDR3: Few seconds 

"""
function MNIST2json()

   ######################
   ### READ TRAINING DATA
   ######################
   train_images = open("/Users/kalyani/0_DATA/MNIST/train-images-idx3-ubyte", "r");
   train_labels = open("/Users/kalyani/0_DATA/MNIST/train-labels-idx1-ubyte", "r");

   # Read header of images file
   # `bswap` ensures integers are read in Big Endian format
   magic_number_f = Int(bswap(read(train_images, UInt32)));
   num_images     = Int(bswap(read(train_images, UInt32)));
   num_rows       = Int(bswap(read(train_images, UInt32)));
   num_cols       = Int(bswap(read(train_images, UInt32)));

   # Read header of labels file
   magic_number_l = Int(bswap(read(train_labels, UInt32)));
   num_labels     = Int(bswap(read(train_labels, UInt32)));

   # Initialize array `features` of pixel values and array `labels`
   # of numbers depicted in images
   # Pixels should be are arranged row-wise.
   # Pixel values range from 0 to 255. 0 means background (white),
   # 255 means foreground (black).
   # Label values range from 0 to 9.
   train_features = Array{UInt8}(undef, num_rows*num_cols, num_images);
   for index in 1:num_images
      train_features[:, index] = convert(Array{UInt8},
                                         read(train_images, num_rows*num_cols));
   end
   train_labels = convert(Array{UInt8}, read(train_labels, num_labels));

   close(train_images);
   close(train_images);

   path_dict(features, labels) =  Dict( "X" => Dict( "data" => features,
                                        "type" => string(typeof(features[1])) ),
                                        "y" => Dict( "data" => labels,
                                        "type" => string(typeof(labels[1])) )
                                      );
   write("MNIST_train.json", JSON.json(path_dict(train_features, train_labels)))

   ##################
   ### READ TEST DATA
   ##################
   test_images = open("/Users/kalyani/0_DATA/MNIST/t10k-images-idx3-ubyte", "r");
   test_labels = open("/Users/kalyani/0_DATA/MNIST/t10k-labels-idx1-ubyte", "r");

   # Read header of images file
   # `bswap` ensures integers are read in Big Endian format
   magic_number_f = Int(bswap(read(test_images, UInt32)));
   num_images     = Int(bswap(read(test_images, UInt32)));
   num_rows       = Int(bswap(read(test_images, UInt32)));
   num_cols       = Int(bswap(read(test_images, UInt32)));

   # Read header of labels file
   magic_number_l = Int(bswap(read(test_labels, UInt32)));
   num_labels     = Int(bswap(read(test_labels, UInt32)));

   # Initialize arrays `features`(which stores pixel values) and `labels`
   # (which stores number depicted in image)
   # Pixels should be are organized row-wise.
   # Pixel values are 0 to 255. 0 means background (white),
   # 255 means foreground (black).
   # Label values are 0 to 9.
   test_features = Array{UInt8}(undef, num_rows*num_cols, num_images);
   for index in 1:num_images
      test_features[:, index] = convert(Array{UInt8},
                                        read(test_images, num_rows*num_cols));
   end
   test_labels = convert(Array{UInt8}, read(test_labels, num_labels));

   close(test_images);
   close(test_images);

   write("MNIST_test.json", JSON.json(path_dict(test_features, test_labels)))
end


"""
    dispMNISTPatches(img, patchSize)

Function to display a grid of `patchSize` X `patchSize` 
images that are chosen randomly from the collection `img`. 

Each row of array `img` represents an image
whose pixel values are stored in column-major format. 
"""
function dispMNISTPatches(img, patchSize)
    all_img = reshape(Float64.(ones(28 * patchSize + patchSize - 1)), :, 1)
    for j = 1:patchSize
        # Rewrite first_n with next n columns
        first_n = convert(Array, vec(img[(j-1)*patchSize+1:j*patchSize, :]'))
        for i = 1:patchSize-1
            # place a row of black pixels between consecutive images
            splice!(
                first_n,
                (28^2)*i + 1 + 28*(i - 1) : (28^2)*i + 28*(i-1),
                Float64.(ones(28))
            )
        end
        first_n = permutedims(reshape(first_n, 28, :), (2, 1))
        # `reshape` rearranges values in column-major order,
        # whereas image pixels are read in row-major order

        all_img = hcat(
            all_img,
            first_n,
            reshape(Float64.(ones(28 * patchSize + patchSize - 1)), :, 1)
        )
    end
    return imPatch = colorview(Gray, Float16.(all_img))
end
