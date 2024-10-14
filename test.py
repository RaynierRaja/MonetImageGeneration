import torch
from models import Generator
from dataset import imshow, ImageDataset, get_data_loaders

# Check if a GPU is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loading the saved model and optimizer states
checkpoint = torch.load("D:\\Personal Projects\\AIProjects\\MonetImageGeneration\\models\\model_state_dict.pt")

# Photo-to-Monet image transformations
photo2monet = Generator()  # Transforms photo-like images to Monet-style images
photo2monet = photo2monet.to(device)  # Photo-to-Monet generator

# Load the state dictionaries into the models
photo2monet.load_state_dict(checkpoint['photo2monet_state_dict'])

photo_datset = ImageDataset(path_dir="D:\\Personal Projects\AIProjects\\MonetImageGeneration\\dataset\\photo_jpg")
photo_dataloader = get_data_loaders(dataset=photo_datset,shuffle=True,batch_size=8)

# Move the test image to the same device as the generated image
test_photo_img = next(iter(photo_dataloader)).to(device)
# Generate the Monet-style image
gen_monet_img = photo2monet(test_photo_img)
# Concatenate the original and generated images horizontally (dim=3 for width concatenation)
concat_img = torch.cat((test_photo_img, gen_monet_img), dim=3)
# Detach the concatenated image and move it back to CPU for visualization
concat_img = concat_img[0].detach().cpu()
# Show the concatenated image
imshow(concat_img)
