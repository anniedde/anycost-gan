# get edit directions for FFHQ models
import sys
sys.path.append('../')
sys.path.append('.')  # to run from the project root dir
import torch
import torch.nn.functional as F
import numpy as np
import models
from tqdm import tqdm
import dnnlib
from legacy import load_network_pkl
from thirdparty.eg3d.camera_utils import LookAtPoseSampler
import os

# configurations for the job
device = 'cuda'
# specify the attributes to compute latent direction
chosen_attr = ['Smiling', 'Young', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Eyeglasses', 'Mustache']
attr_list = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips',
             'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby',
             'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male',
             'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose',
             'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair',
             'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']
space = 'w'  # chosen from ['z', 'w', 'w+']
config = 'anycost-ffhq-config-f'

@torch.no_grad()
def get_style_attribute_pairs_celeb(celeb, model):  # this function is written with horovod to accelerate the extraction (by n_gpu times)

    torch.manual_seed(0)
    print(' * Extracting style-attribute pairs...')
    # build and load the pre-trained attribute predictor on CelebA-HQ
    predictor = models.get_pretrained('attribute-predictor').to(device)
    # build and load the pre-trained eg3d generator
    network_pkl = f'/playpen-nas-ssd/awang/semantic_editing/networks/{celeb}/{model}.pkl'
    print('Loading trained eg3d generator network from "%s"...' % network_pkl)
    with dnnlib.util.open_url(network_pkl) as f:
        generator = load_network_pkl(f)['G_ema'].to(device) # type: ignore

    predictor.eval()
    generator.eval()

    # randomly generate images and feed them to the predictor
    # configs from https://github.com/genforce/interfacegan
    randomized_noise = False
    truncation_psi = 0.7
    batch_size = 20
    n_batch = 500000 // (batch_size)
    cam2world_pose = LookAtPoseSampler.sample(3.14/2, 3.14/2, torch.tensor([0, 0, 0], device=device), radius=2.7, device=device)
    intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=device)
    cam_pivot = torch.tensor(generator.rendering_kwargs.get('avg_camera_pivot', [0, 0, 0]), device=device)
    cam_radius = generator.rendering_kwargs.get('avg_camera_radius', 2.7)
    conditioning_cam2world_pose = LookAtPoseSampler.sample(np.pi/2, np.pi/2, cam_pivot, radius=cam_radius, device=device)
    conditioning_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1).repeat(batch_size, 1)  
    camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1).repeat(batch_size, 1).to(device)
    
    styles = []
    attributes = []

    #mean_style = generator.mean_style(100000).view(1, 1, -1)
    assert space in ['w', 'w+', 'z']
    for i in tqdm(range(n_batch)):
        print(i)
        z = torch.randn(batch_size, generator.z_dim, device=device)
        #print('size of z: ', z.size())
        #print('size of c: ', conditioning_params.size())

        w = generator.mapping(z, conditioning_params, truncation_psi=truncation_psi, truncation_cutoff=14)
        #print('size of w: ', w.size())
        images = generator.synthesis(w, camera_params, noise_mode='const')['image']#.detach()
        #print('size of images: ', images.size())
        
        images = F.interpolate(images.clamp(-1, 1), size=256, mode='bilinear', align_corners=True)
        attr = predictor(images)
        #print('attr size: ', attr.size())
        # move to cpu to save memory
        if space == 'w+':
            styles.append(w.to('cpu'))
        elif space == 'w':
            #styles.append(w.to('cpu'))
            styles.append(w.mean(1, keepdim=True).to('cpu'))  # originally duplicated
        else:
            styles.append(z.to('cpu'))
        attributes.append(attr.to('cpu'))

    styles = torch.cat(styles, dim=0)
    attributes = torch.cat(attributes, dim=0)
    print('attributes size: ', attributes.size())
    print('styles size: ', styles.size())

    save_loc = f'celeb_attributes_styles/{celeb}_{model}'
    if not os.path.isdir(save_loc):
        os.mkdir(save_loc)
    torch.save(attributes, '{}/attributes.pt'.format(save_loc))
    torch.save(styles, '{}/styles.pt'.format(save_loc))

def extract_boundaries_celeb(celeb, model):
    styles = torch.load(f'celeb_attributes_styles/{celeb}_{model}/styles.pt')
    attributes = torch.load(f'celeb_attributes_styles/{celeb}_{model}/attributes.pt')
    attributes = attributes.view(-1, 40, 2)
    prob = F.softmax(attributes, dim=-1)[:, :, 1]  # probability to be positive [n, 40]

    boundaries = {}
    for idx, attr in tqdm(enumerate(attr_list), total=len(attr_list)):
        this_prob = prob[:, idx]

        from thirdparty.manipulator import train_boundary
        boundary = train_boundary(latent_codes=styles.squeeze().cpu().numpy(),
                                  scores=this_prob.view(-1, 1).cpu().numpy(),
                                  chosen_num_or_ratio=0.02,
                                  split_ratio=0.7,
                                  )
        key_name = '{:02d}'.format(idx) + '_' + attr
        boundaries[key_name] = boundary

    boundaries = {k: torch.tensor(v) for k, v in boundaries.items()}
    torch.save(boundaries, f'celeb_attributes_styles/{celeb}_{model}/boundaries.pt')

@torch.no_grad()
def get_style_attribute_pairs(network_pkl):  # this function is written with horovod to accelerate the extraction (by n_gpu times)

    torch.manual_seed(0)
    print(' * Extracting style-attribute pairs...')
    # build and load the pre-trained attribute predictor on CelebA-HQ
    predictor = models.get_pretrained('attribute-predictor').to(device)
    # build and load the pre-trained eg3d generator
    print('Loading trained eg3d generator network from "%s"...' % network_pkl)
    with dnnlib.util.open_url(network_pkl) as f:
        generator = load_network_pkl(f)['G_ema'].to(device) # type: ignore

    predictor.eval()
    generator.eval()

    # randomly generate images and feed them to the predictor
    # configs from https://github.com/genforce/interfacegan
    randomized_noise = False
    truncation_psi = 0.7
    batch_size = 2
    n_batch = 500000 // (batch_size)
    cam2world_pose = LookAtPoseSampler.sample(3.14/2, 3.14/2, torch.tensor([0, 0, 0], device=device), radius=2.7, device=device)
    intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=device)
    cam_pivot = torch.tensor(generator.rendering_kwargs.get('avg_camera_pivot', [0, 0, 0]), device=device)
    cam_radius = generator.rendering_kwargs.get('avg_camera_radius', 2.7)
    conditioning_cam2world_pose = LookAtPoseSampler.sample(np.pi/2, np.pi/2, cam_pivot, radius=cam_radius, device=device)
    conditioning_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)        
    camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1).to(device)
    
    styles = []
    attributes = []

    #mean_style = generator.mean_style(100000).view(1, 1, -1)
    assert space in ['w', 'w+', 'z']
    for i in tqdm(range(n_batch)):
        print(i)
        z = torch.randn(1, generator.z_dim, device=device)

        w = generator.mapping(z, conditioning_params, truncation_psi=truncation_psi, truncation_cutoff=14)
        images = generator.synthesis(w, camera_params, noise_mode='const')['image']#.detach()
        
        images = F.interpolate(images.clamp(-1, 1), size=256, mode='bilinear', align_corners=True)
        attr = predictor(images)
        # move to cpu to save memory
        if space == 'w+':
            styles.append(w.to('cpu'))
        elif space == 'w':
            styles.append(w.mean(1, keepdim=True).to('cpu'))  # originally duplicated
        else:
            styles.append(z.to('cpu'))
        attributes.append(attr.to('cpu'))

    styles = torch.cat(styles, dim=0)
    print('styles size: ', styles.size())
    attributes = torch.cat(attributes, dim=0)

    torch.save(attributes, 'attributes_{}.pt'.format(config))
    torch.save(styles, 'styles_{}.pt'.format(config))


def extract_boundaries():
    styles = torch.load('styles_{}.pt'.format(config))
    attributes = torch.load('attributes_{}.pt'.format(config))
    attributes = attributes.view(-1, 40, 2)
    prob = F.softmax(attributes, dim=-1)[:, :, 1]  # probability to be positive [n, 40]

    boundaries = {}
    for idx, attr in tqdm(enumerate(attr_list), total=len(attr_list)):
        this_prob = prob[:, idx]

        from thirdparty.manipulator import train_boundary
        boundary = train_boundary(latent_codes=styles.squeeze().cpu().numpy(),
                                  scores=this_prob.view(-1, 1).cpu().numpy(),
                                  chosen_num_or_ratio=0.02,
                                  split_ratio=0.7,
                                  )
        key_name = '{:02d}'.format(idx) + '_' + attr
        boundaries[key_name] = boundary

    boundaries = {k: torch.tensor(v) for k, v in boundaries.items()}
    torch.save(boundaries, 'boundaries_{}.pt'.format(config))


# experimental; not yet used in the demo
# do not observe significant improvement right now
def project_boundaries():  # only project the ones used for demo
    from thirdparty.manipulator import project_boundary
    boundaries = torch.load('boundaries_{}.pt'.format(config))
    chosen_idx = [attr_list.index(attr) for attr in chosen_attr]
    sorted_keys = ['{:02d}'.format(idx) + '_' + attr_list[idx] for idx in chosen_idx]
    all_boundaries = np.concatenate([boundaries[k].cpu().numpy() for k in sorted_keys])  # n, 512
    similarity = all_boundaries @ all_boundaries.T
    projected_boundaries = []
    for i_b in range(len(sorted_keys)):
        # NOTE: the number of conditions is exponential;
        # we only take the 2 most related boundaries
        this_sim = similarity[i_b]
        this_sim[i_b] = -100.  # exclude self
        idx1, idx2 = np.argsort(this_sim)[-2:]  # most similar 2
        projected_boundaries.append(project_boundary(all_boundaries[i_b][None],
                                                     all_boundaries[idx1][None], all_boundaries[idx2][None]))
    boundaries = {k: v for k, v in zip(sorted_keys, torch.tensor(projected_boundaries))}
    torch.save(boundaries, 'boundary_projected_{}.pt'.format(config))


if __name__ == '__main__':
    network_pkl = '/playpen-nas-ssd/awang/semantic_editing/mystyle/trained_models/barack_no_lora.pkl'

    get_style_attribute_pairs(network_pkl)
    #extract_boundaries()
    # project_boundaries()
