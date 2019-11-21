# Thanks for the open source code
# https://github.com/kimvwijnen/geodesic_distance_transform
import numpy as np


def get_seeds(mask):
    seeds = np.array(np.where(mask == 1)).T
    return seeds


def checkdots_anydim(dist, seedarray):
    for s in seedarray:
        assert (dist[tuple(s)] == 0.0), \
            '***BUG***! distance of dot is larger than 0! Distance is: ' + \
            str(dist[tuple(s)])


def calc_edm(alpha, p_val, q_val, scaling_factor, squared_dist_f, i):
    # calculate spatial distance
    # (part of gdm formula without intensity, so basically just euclidean distance)
    dist_pq = alpha * np.sqrt(squared_dist_f[i])
    return dist_pq


def calc_idm(alpha, p_val, q_val, scaling_factor, squared_dist_f, i):
    # calculate intensity distance
    # (part of gdm formula without spatial distance, so basically absolute intensity difference)
    dist_pq = alpha * np.sqrt(np.square(p_val - q_val) * scaling_factor)
    return dist_pq


def calc_gdm(alpha, p_val, q_val, scaling_factor, squared_dist_f, i):
    # calculate geodesic distance, combines spatial and intensity distance in image
    # in paper: dq = alpha*sqrt((G(p)-G(q))^2 + beta)
    dist_pq = alpha * np.sqrt(np.square(p_val - q_val)
                              * scaling_factor + squared_dist_f[i])
    return dist_pq


def pass_3D(img, distance, k_p, func_dist, squared_dist, scaling_factor, alpha=1.0, backward=False):
    width = img.shape[0]
    height = img.shape[1]
    depth = img.shape[2]

    if backward:
        # loop through image in reverse order
        loopd = np.arange(depth)[::-1]
        looph = np.arange(height)[::-1]
        loopw = np.arange(width)[::-1]
    else:
        # loop through image in forward order
        loopd = np.arange(depth)
        looph = np.arange(height)
        loopw = np.arange(width)

    # pass through image
    for d in loopd:
        for h in looph:
            for w in loopw:

                # get distance and intensity value at point p
                p_dist = distance[w, h, d]
                p_val = img[w, h, d]

                # looping through kernel
                for i in range(k_p['h'].shape[0]):
                    nd = d + k_p['d'][i]
                    nh = h + k_p['h'][i]
                    nw = w + k_p['w'][i]

                    # if selected pixel not in image continue
                    if nd < 0 or nd >= depth or nh < 0 or nh >= height or nw < 0 or nw >= width:
                        continue

                    # distance and intensity value at point q of the kernel
                    q_dist = distance[nw, nh, nd]
                    q_val = img[nw, nh, nd]

                    # compute distance between voxel p and q
                    dist_pq = func_dist(alpha, p_val, q_val,
                                        scaling_factor, squared_dist, i)

                    # add distance between voxel p and voxel q to distance value in q
                    # gives full distance to a seed point
                    # Formula:
                    # F*(p) = min(F(p),calc_distances) with calc_dist = 1+dq+F*(q) for all places in kernel
                    calc_dist = dist_pq + q_dist

                    # select minimal distance of calculated distance and current distance
                    # for point p
                    if calc_dist < p_dist:
                        p_dist = calc_dist

                # after going through kernel the selected distance is added to the distance map
                distance[w, h, d] = p_dist

    return distance


def pass_2D(img, distance, k_p, func_dist, squared_dist, scaling_factor, alpha=1.0, backward=False):
    width = img.shape[0]
    height = img.shape[1]

    if backward:
        # loop through image in reverse order
        looph = np.arange(height)[::-1]
        loopw = np.arange(width)[::-1]
    else:
        # loop through image in forward order
        looph = np.arange(height)
        loopw = np.arange(width)

    # pass through image
    for h in looph:
        for w in loopw:

            # get distance and intensity value at point p
            p_dist = distance[w, h]
            p_val = img[w, h]

            # looping through kernel
            for i in range(k_p['h'].shape[0]):
                nh = h + k_p['h'][i]
                nw = w + k_p['w'][i]

                # if selected pixel not in image continue
                if nh < 0 or nh >= height or nw < 0 or nw >= width:
                    continue

                # distance and intensity value at point q
                q_dist = distance[nw, nh]
                q_val = img[nw, nh]

                # compute distance between voxel p and q
                dist_pq = func_dist(alpha, p_val, q_val,
                                    scaling_factor, squared_dist, i)

                # add distance between pixel p and pixel q to distance value in q
                # gives full distance to a seed point
                # Formula:
                # F*(p) = min(F(p),calc_distances) with calc_dist = 1+dq+F*(q) for all places in kernel
                calc_dist = dist_pq + q_dist

                # select minimal distance of calculated distance and current distance
                # for point p
                if calc_dist < p_dist:
                    p_dist = calc_dist

            # after going through kernel the selected distance is added to the distance map
            distance[w, h] = p_dist

    return distance


def get_3d_kernel(backward=False):
    if backward:
        # kernel for the backward scan (paper Toivanen, table 4)
        kd_b = np.array([-1, -1, -1, -1, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        kh_b = np.array([0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1])
        kw_b = np.array([1, -1, 0, 1, 1, -1, 0, 1, 0, 1, -1, 0, 1])

        k_p = {'d': kd_b, 'h': kh_b, 'w': kw_b}

        # squared Euclidean distance between point p to point q (point in kernel)
        # for the voxels in the backward kernel
        squared_dist = np.array(
            [2.0, 3.0, 2.0, 3.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 3.0, 2.0, 3.0])

    else:
        # kernel for the forward scan (paper Toivanen, table 3)
        kd_f = np.array([-1, -1, -1, -1, -1, 0, 0, 0, 0, 1, 1, 1, 1])
        kh_f = np.array([-1, -1, -1, 0, 0, -1, -1, -1, 0, -1, -1, -1, 0])
        kw_f = np.array([-1, 0, 1, -1, 0, -1, 0, 1, -1, -1, 0, 1, -1])

        k_p = {'d': kd_f, 'h': kh_f, 'w': kw_f}
        # squared Euclidean distance between point p to point q (point in kernel)
        # for the voxels in the forward kernel
        squared_dist = np.array(
            [3.0, 2.0, 3.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 3.0, 2.0, 3.0, 2.0])

    return k_p, squared_dist


def get_2d_kernel(backward=False):
    if backward:
        # kernel for the backward scan (paper Toivanen, table 4)
        kh_b = np.array([0, 1, 1, 1])
        kw_b = np.array([1, -1, 0, 1])

        k_p = {'h': kh_b, 'w': kw_b}

        # squared Euclidean distance between point p to point q (point in kernel)
        # for the pixels in the backward kernel
        squared_dist = np.array([1.0, 2.0, 1.0, 2.0])

    else:
        # kernel for the forward scan (paper Toivanen, table 3)
        kh_f = np.array([-1, -1, -1, 0])
        kw_f = np.array([-1, 0, 1, -1])

        k_p = {'h': kh_f, 'w': kw_f}

        # squared Euclidean distance between point p to point q (point in kernel)
        # for the pixels in the forward kernel
        squared_dist = np.array([2.0, 1.0, 2.0, 1.0])

    return k_p, squared_dist


def compute_dm_rasterscan(im, seeds, its=8, dist_type='', scaling_factor=1):
    # distance map raster scan
    # following paper from Toivanen et al
    # scaling_factor: change the weighting of euclidean distance vs intensity distance

    # im is a 2D or 3D image with dimensions width, height(, depth)
    # shape of image defines if the distance map will be 2D or 3D
    # seeds is a 2D array with dimensions #seeds by 2 or 3 (xy(z) coords)
    # Image is not transposed, dimensions are width x height x depth

    img = np.squeeze(np.array(im, dtype=np.float32))
    seeds = np.array(seeds, dtype=np.int32)

    assert seeds.shape[1] == len(img.shape), \
        "ERROR, intensity image " + \
        str(len(img.shape)) + "D but seed points " + \
        str(seeds.shape[1]) + "D, should be the same!"

    if len(img.shape) > 2:
        # 3D distance map
        dim_dm = 3
        get_kernel = get_3d_kernel
        pass_func = pass_3D
    else:
        # 2D distance map
        dim_dm = 2
        get_kernel = get_2d_kernel
        pass_func = pass_2D

    print('Computing ' + dist_type + ' distance map in ' + str(dim_dm) + 'D')

    # Define distance function to be used for computing the distance map
    if 'euclidean' in dist_type:
        func_dist = calc_edm
    elif ('gradient' in dist_type) or ('intensity' in dist_type):
        func_dist = calc_idm
    elif 'geodesic' in dist_type:
        func_dist = calc_gdm
    else:
        print('***WARNING*** Using geodesic distance because no (correct) distance measure specified!!')
        func_dist = calc_gdm

    # setting all initial distances at a high value except the seeds..
    distance = 1.0e10 * np.ones(img.shape, dtype=np.float32)

    # ..the seeds are set to 0
    for s in seeds:
        distance[tuple(s)] = 0.0

    for it in range(its):
        # forward scan
        k_f, squared_dist_f = get_kernel()
        distance = pass_func(img, distance, k_f, func_dist,
                             squared_dist_f, scaling_factor)

        # backward scan
        k_b, squared_dist_b = get_kernel(backward=True)
        distance = pass_func(img, distance, k_b, func_dist,
                             squared_dist_b, scaling_factor, backward=True)

        # sanity check:
        # check if the distance in any of the seeds is not 0 anymore
        # would mean something is wrong
        checkdots_anydim(distance, seeds)

    return distance


def compute_mp(distance, p=5):
    mp = np.power((1 - (distance) / (np.max(distance))), p)
    return mp
