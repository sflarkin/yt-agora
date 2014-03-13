from yt.testing import *

def test_ray():
    for nproc in [1, 2, 4, 8]:
        pf = fake_random_pf(64, nprocs=nproc)
        dx = (pf.domain_right_edge - pf.domain_left_edge) / \
          pf.domain_dimensions
        # Three we choose, to get varying vectors, and ten random
        pp1 = np.random.random((3, 13))
        pp2 = np.random.random((3, 13))
        pp1[:,0] = [0.1, 0.2, 0.3]
        pp2[:,0] = [0.8, 0.1, 0.4]
        pp1[:,1] = [0.9, 0.2, 0.3]
        pp2[:,1] = [0.8, 0.1, 0.4]
        pp1[:,2] = [0.9, 0.2, 0.9]
        pp2[:,2] = [0.8, 0.1, 0.4]
        unitary = pf.arr(1.0, '')
        for i in range(pp1.shape[1]):
            p1 = pf.arr(pp1[:,i] + 1e-8 * np.random.random(3), 'code_length')
            p2 = pf.arr(pp2[:,i] + 1e-8 * np.random.random(3), 'code_length')

            my_ray = pf.ray(p1, p2)
            yield assert_rel_equal, my_ray['dts'].sum(), unitary, 14
            ray_cells = my_ray['dts'] > 0

            # find cells intersected by the ray
            my_all = pf.h.all_data()
            
            dt = np.abs(dx / (p2 - p1))
            tin  = uconcatenate([[(my_all['x'] - p1[0]) / (p2 - p1)[0] - 0.5 * dt[0]],
                                 [(my_all['y'] - p1[1]) / (p2 - p1)[1] - 0.5 * dt[1]],
                                 [(my_all['z'] - p1[2]) / (p2 - p1)[2] - 0.5 * dt[2]]])
            tout = uconcatenate([[(my_all['x'] - p1[0]) / (p2 - p1)[0] + 0.5 * dt[0]],
                                 [(my_all['y'] - p1[1]) / (p2 - p1)[1] + 0.5 * dt[1]],
                                 [(my_all['z'] - p1[2]) / (p2 - p1)[2] + 0.5 * dt[2]]])
            tin = tin.max(axis=0)
            tout = tout.min(axis=0)
            my_cells = (tin < tout) & (tin < 1) & (tout > 0)
            dts = np.clip(tout[my_cells], 0.0, 1.0) - np.clip(tin[my_cells], 0.0, 1.0)

            yield assert_equal, ray_cells.sum(), my_cells.sum()
            yield assert_rel_equal, my_ray['density'][ray_cells].sum(), \
                                    my_all['density'][my_cells].sum(), 14
            yield assert_rel_equal, my_ray['dts'].sum(), unitary, 14
