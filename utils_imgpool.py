"""Image pool buffer for preventing GAN mode collapse
From https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/2215995e70b16e3f8f6cc1e430dedb3ebe4e110f/util/image_pool.py
"""

import random
import torch
import os


class ImagePool():
    """This class implements an image buffer that stores previously generated images.
    This buffer enables us to update discriminators using a history of generated images
    rather than the ones produced by the latest generators.
    """

    def __init__(self, pool_size, num_class, save_dir=False, resume_buffer=False, device='cuda'):
        """Initialize the ImagePool class
        Parameters:
            pool_size (int) -- the size of image buffer, if pool_size=0, no buffer will be created
        """
        self.pool_size = pool_size
        self.save_dir = save_dir
        self.device = device
        self.num_class = num_class
        if self.pool_size > 0:  # create an empty pool
            if (not self.save_dir) or (not os.path.exists(os.path.join(self.save_dir, "ImagePool.pt"))) or (not resume_buffer):
                self.num_imgs = torch.zeros(self.num_class)
                self.images = [[]] * self.num_class
                # create folder for saving
                os.makedirs(self.save_dir, exist_ok=True)
                print("Preparing folder {} for ImagePool...".format(self.save_dir))
            else:
                
                self.load()
            
      
    def query(self, images, targets):
        """Return an image from the pool.
        Parameters:
            images: the latest generated images from the generator
            targets: the conditional class labels
        Returns images from the buffer.
        By 50/100, the buffer will return input images.
        By 50/100, the buffer will return images previously stored in the buffer,
        and insert the current images to the buffer.

        Note all images are stored in cpu and convert to GPU tensor when sampled
        Use it under torch.no_grad otherwise it will be a mess
        """
        images = images.detach()
        targets = targets.detach()
        if self.pool_size == 0:  # if the buffer size is 0, do nothing
            return images
        return_images = []
        for index, image in enumerate(images):
            image = torch.unsqueeze(image.data, 0)
            target = targets[index] # get target class number for this image
            if self.num_imgs[target] < self.pool_size:   # if the buffer is not full; keep inserting current images to the buffer
                self.num_imgs[target] = self.num_imgs[target] + 1
                # add to img pool
                self.images[target].append(image.cpu())
                return_images.append(image) # still in device
                # print("pool not full, store images for class {} (now {}%)".format(target, self.num_imgs[target] / self.pool_size * 100))
            else:
                p = random.uniform(0, 1)
                if p > 0.5:  # by 50% chance, the buffer will return a previously stored image, and insert the current image into the buffer
                    random_id = random.randint(0, self.num_imgs[target] - 1)  # randint is inclusive
                    tmp = self.images[target][random_id].clone()
                    self.images[target][random_id] = image.cpu() # insert new images
                    # print("select img from pool class {} (poolsize {})".format(target, self.num_imgs[target]))
                    return_images.append(tmp.to(self.device))
                else:       # by another 50% chance, the buffer will return the current image
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)   # collect all the images and return

        self.capacity = (self.num_imgs / self.pool_size).mean()
        return return_images


    def save(self):
        """save method for saving the image pool"""
        assert self.save_dir, "Save dir is 'False'... {}".format(self.save_dir)
        print("Saving Buffer to {}...".format(os.path.join(self.save_dir, "ImagePool.pt")))
        torch.save([self.images, self.num_imgs], os.path.join(self.save_dir, "ImagePool.pt"))
        print("Buffer saved. Buffer capacity {} %!".format( self.capacity * 100 ))
    
    def load(self):
        self.images, self.num_imgs = torch.load(os.path.join(self.save_dir, "ImagePool.pt"))
        self.capacity = (self.num_imgs / self.pool_size).mean()
        print("Resume Buffer from folder {} for ImagePool (Buffer capacity {:.3f} %)...".format(os.path.join(self.save_dir, "ImagePool.pt"), self.capacity * 100))
        

