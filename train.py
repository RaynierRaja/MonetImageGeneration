import torch

from models import Generator, Discriminator, display
from training import generator_loss, discriminator_loss, cycle_consistency_loss, identity_loss
from dataset import imshow, ImageDataset, get_data_loaders

monet_datset = ImageDataset(path_dir="D:\\Personal Projects\AIProjects\\MonetImageGeneration\\dataset\\monet_jpg")
monet_dataloader = get_data_loaders(dataset=monet_datset,shuffle=False,batch_size=8)

photo_datset = ImageDataset(path_dir="D:\\Personal Projects\AIProjects\\MonetImageGeneration\\dataset\\photo_jpg")
photo_dataloader = get_data_loaders(dataset=photo_datset,shuffle=False,batch_size=8)

# Check if a GPU is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate generators for Monet-to-Photo and Photo-to-Monet image transformations
monet2photo = Generator()  # Transforms Monet-style images to photo-like images
photo2monet = Generator()  # Transforms photo-like images to Monet-style images

# Instantiate discriminators (classifiers) for distinguishing real vs. fake Monet and photo images
monet_classifier = Discriminator()  # Classifies Monet-style images (real or generated)
photo_classifier = Discriminator()  # Classifies photo-like images (real or generated)

# Move the models to the GPU
monet2photo = monet2photo.to(device)  # Monet-to-Photo generator
photo2monet = photo2monet.to(device)  # Photo-to-Monet generator
monet_classifier = monet_classifier.to(device)  # Monet discriminator
photo_classifier = photo_classifier.to(device)  # Photo discriminator

# Setup the optimizers
monet2photo_optimizer = torch.optim.Adam(monet2photo.parameters(), lr=0.0002, betas=(0.5, 0.999))
photo2monet_optimizer = torch.optim.Adam(photo2monet.parameters(), lr=0.0002, betas=(0.5, 0.999))
monet_classifier_optimizer = torch.optim.Adam(monet_classifier.parameters(), lr=0.0002, betas=(0.5, 0.999))
photo_classifier_optimizer = torch.optim.Adam(photo_classifier.parameters(), lr=0.0002, betas=(0.5, 0.999))

epoch = 40
for i in range(epoch):
    # Sample input Data
    monet_img = next(iter(monet_dataloader)).to(device)
    photo_img = next(iter(photo_dataloader)).to(device)

    # Generate a fake photo by passing a Monet-style image through the Monet-to-Photo generator
    fake_photo = monet2photo(monet_img)  # Monet-to-photo transformation (fake photo)

    # Cycle the fake photo back into a Monet-style image using the Photo-to-Monet generator
    cycled_monets = photo2monet(fake_photo)  # Cycled back to Monet-style (reconstruction)

    # Generate a fake Monet image by passing a photo-like image through the Photo-to-Monet generator
    fake_monet = photo2monet(photo_img)  # Photo-to-Monet transformation (fake Monet)

    # Cycle the fake Monet back into a photo-like image using the Monet-to-Photo generator
    cycled_photos = monet2photo(fake_monet)  # Cycled back to photo-like (reconstruction)

    # Identity mapping for photo images passed through the Monet-to-Photo generator (should produce the same image)
    same_photo = monet2photo(photo_img)  # Identity transformation for photos (consistency check)

    # Identity mapping for Monet images passed through the Photo-to-Monet generator (should produce the same image)
    same_monet = photo2monet(monet_img)  # Identity transformation for Monets (consistency check)

    # Classify the real photo using the photo discriminator (for loss calculation)
    real_photo_pred = photo_classifier(photo_img)  # Discriminator prediction for real photo

    # Classify the fake photo using the photo discriminator (for adversarial loss)
    fake_photo_pred = photo_classifier(fake_photo)  # Discriminator prediction for fake photo

    # Classify the real Monet using the Monet discriminator (for loss calculation)
    real_monet_pred = monet_classifier(monet_img)  # Discriminator prediction for real Monet

    # Classify the fake Monet using the Monet discriminator (for adversarial loss)
    fake_monet_pred = monet_classifier(fake_monet)  # Discriminator prediction for fake Monet

    # Generator loss
    monet_gen_loss = generator_loss(fake_monet_pred)
    photo_gen_loss = generator_loss(fake_photo_pred)

    # Discriminator loss
    photo_disc_loss = discriminator_loss(real_pred=real_photo_pred, gen_pred=fake_photo_pred)
    monet_disc_loss = discriminator_loss(real_pred=real_monet_pred, gen_pred=fake_monet_pred)

    # Cycle loss
    monet_cycle_loss = cycle_consistency_loss(monet_img,cycled_monets)
    photo_cycle_loss = cycle_consistency_loss(photo_img,cycled_photos)

    # Identity loss
    monet_identity_loss = identity_loss(monet_img, same_monet)
    photo_identity_loss = identity_loss(photo_img, same_photo)

    # Combine the generator losses (sum of generator, cycle consistency, and identity losses)
    total_monet_gen_loss = monet_gen_loss + monet_cycle_loss + monet_identity_loss
    total_photo_gen_loss = photo_gen_loss + photo_cycle_loss + photo_identity_loss

    # Combine the discriminator losses
    total_disc_loss = photo_disc_loss + monet_disc_loss

    # Zero the gradients
    monet2photo_optimizer.zero_grad()
    photo2monet_optimizer.zero_grad()
    monet_classifier_optimizer.zero_grad()
    photo_classifier_optimizer.zero_grad()

    # Backpropagate the losses
    total_monet_gen_loss.backward(retain_graph=True)  # Backprop for Monet-to-Photo generator
    total_photo_gen_loss.backward(retain_graph=True)  # Backprop for Photo-to-Monet generator
    total_disc_loss.backward(retain_graph=True)  # Backprop for photo discriminators

    # Update parameters
    monet2photo_optimizer.step()
    photo2monet_optimizer.step()
    monet_classifier_optimizer.step()
    photo_classifier_optimizer.step()

torch.save({
            'monet2photo_state_dict': monet2photo.state_dict(),
            'photo2monet_state_dict': photo2monet.state_dict(),
            'monet_classifier_state_dict': monet_classifier.state_dict(),
            'photo_classifier_state_dict': photo_classifier.state_dict(),
            'monet2photo_optimizer_state_dict': monet2photo_optimizer.state_dict(),
            'photo2monet_optimizer_state_dict': photo2monet_optimizer.state_dict(),
            'monet_classifier_optimizer_state_dict': monet_classifier_optimizer.state_dict(),
            'photo_classifier_optimizer_state_dict': photo_classifier_optimizer.state_dict(),
            }, "D:\\Personal Projects\\AIProjects\\MonetImageGeneration\\models\\model_state_dict.pt")

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
