import os
import torch
import numpy as np
import imageio
import json
import torch.nn.functional as F
import cv2

def load_audface_data(basedir, testskip=1, test_file=None, aud_file=None):
    if test_file is not None:
        with open(os.path.join(basedir, test_file)) as fp:
            meta = json.load(fp)
        poses = []
        auds = []
        aud_features = np.load(os.path.join(basedir, aud_file))

        for frame in meta['frames'][::testskip]: #
            poses.append(np.array(frame['transform_matrix']))
            auds.append(aud_features[min(frame['aud_id'], aud_features.shape[0]-1)]) #'frame_id' -> 'img_id'
        poses = np.array(poses).astype(np.float32)
        auds = np.array(auds).astype(np.float32)
        bc_img = cv2.imread(os.path.join(basedir, 'bc.jpg'))
        bc_img = cv2.cvtColor(bc_img, cv2.COLOR_BGR2RGB)
        H, W = bc_img.shape[0], bc_img.shape[1]
        focal, cx, cy = float(meta['focal_len']), float(meta['cx']), float(meta['cy'])
        return poses, auds, bc_img, [H, W, focal, cx, cy]

    splits = ['train', 'val']
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)
    all_imgs = []
    all_poses = []
    all_auds = []
    all_sample_rects = []
    aud_features = np.load(os.path.join(basedir, aud_file))
    counts = [0]
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        auds = []
        sample_rects = []
        mouth_rects = []
        #exps = []
        if s == 'train' or testskip == 0:
            skip = 1
        else:
            skip = testskip
        for frame in meta['frames'][::skip]: #
            fname = os.path.join(basedir, 'head_imgs', str(frame['img_id']) + '.jpg')
            imgs.append(fname)
            poses.append(np.array(frame['transform_matrix']))
            auds.append(aud_features[min(frame['aud_id'], aud_features.shape[0]-1)])
            sample_rects.append(np.array(frame['face_rect'], dtype=np.int32))
        imgs = np.array(imgs)
        poses = np.array(poses).astype(np.float32)
        auds = np.array(auds).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)
        all_auds.append(auds)
        all_sample_rects.append(sample_rects)

    i_split = [np.arange(counts[i], counts[i+1]) for i in range(len(splits))]
    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    auds = np.concatenate(all_auds, 0)
    sample_rects = np.concatenate(all_sample_rects, 0)

    bc_img = cv2.imread(os.path.join(basedir, 'bc.jpg'))
    bc_img = cv2.cvtColor(bc_img, cv2.COLOR_BGR2RGB)

    H, W = bc_img.shape[:2]
    focal, cx, cy = float(meta['focal_len']), float(
        meta['cx']), float(meta['cy'])

    return imgs, poses, auds, bc_img, [int(H), int(W), focal, cx, cy], sample_rects, sample_rects, i_split

def get_rays_np(H, W, focal, cx, cy, c2w):
    if cx is None:
        cx = W*.5
    if cy is None:
        cy = H*.5
    i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-cx)/focal, -(j-cy)/focal, -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))
    return rays_o, rays_d

def raw2outputs(raw, z_vals, rays_d, bc_rgb, raw_noise_std=0, white_bkgd=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    def raw2alpha(raw, dists, act_fn=F.relu): return 1. - \
        torch.exp(-(act_fn(raw)+1e-6)*dists)

    dists = z_vals[..., 1:] - z_vals[..., :-1]
    temp = torch.Tensor([1e10]).to(dists.device) #
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(
        dists[..., :1].shape).to(dists.device)], -1)  # [B, N_rays, N_samples] #device

    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    rgb = torch.sigmoid(raw[..., :3])  # [B, N_rays, N_samples, 3]
    rgb = torch.cat((rgb[..., :-1, :], bc_rgb.unsqueeze(-2)), dim=-2)
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[..., 3].shape) * raw_noise_std
        noise = noise.to(raw.device)

    # [B*N_rays, N_samples] 
    alpha = raw2alpha(raw[..., 3] + noise, dists)  

    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    temp = torch.ones((alpha.shape[0], alpha.shape[1], 1)).to(alpha.device)
    weights = alpha * \
        torch.cumprod(torch.cat([temp, 1.-alpha + 1e-10], -1), -1)[..., :-1]
    rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_imgs, N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map),
                            depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[..., None])

    return rgb_map, disp_map, acc_map, weights, depth_map

def sample_pdf(bins, weights, N_samples, det=False):
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    # (batch, len(bins))
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    u = u.to(weights.device)
    cdf = cdf.to(weights.device)
    bins = bins.to(weights.device)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], inds_g.shape[2], cdf.shape[-1]]
    # print(inds_g.shape)
    # print(matched_shape)
    # print(cdf.shape)
    cdf_g = torch.gather(cdf.unsqueeze(2).expand(matched_shape), 3, inds_g)
    bins_g = torch.gather(bins.unsqueeze(2).expand(matched_shape), 3, inds_g)

    denom = (cdf_g[..., 1]-cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[..., 0])/denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1]-bins_g[..., 0])
    # print(samples.shape)
    return samples

def update_n_append_dict(x, y, output_to_cpu=False):
    """
    update y with x
    if x in y append
    """
    for key in x:
        item = x[key]
        if output_to_cpu:
            item = item.cpu()
        if key in y:
            y[key].append(item)
        else:
            y[key] = [item,]

def concat_all_items_in_dict(x):
    y = {}
    for key in x:
        y[key] = torch.cat(x[key], 1)
    return y