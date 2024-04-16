import numpy as onp
import jax
import jax.numpy as np
import matplotlib.pyplot as plt
import meshio
import datetime

from jax_ldpm.utils import json_parse
from jax_ldpm.generate_mesh import box_mesh, save_sol
from jax_ldpm.core import *


onp.set_printoptions(threshold=sys.maxsize,
                     linewidth=1000,
                     suppress=True,
                     precision=5)


crt_dir = os.path.dirname(__file__)
input_dir = os.path.join(crt_dir, 'input')
output_dir = os.path.join(crt_dir, 'output')
vtk_dir = os.path.join(output_dir, 'vtk')
numpy_dir = os.path.join(output_dir, 'numpy')
freecad_dir = os.path.join(input_dir, 'freecad/50x50x50')


def test():
    facet_data = onp.genfromtxt(os.path.join(freecad_dir, 'LDPMgeo000-data-facets.dat'), dtype=float)
    facet_vertices = onp.genfromtxt(os.path.join(freecad_dir, 'LDPMgeo000-data-facetsVertices.dat'), dtype=float)

    json_file = os.path.join(input_dir, 'json/params.json')
    params = json_parse(json_file)
    params['rho'] = 1.

    freecad_mesh_file = os.path.join(freecad_dir, 'LDPMgeo000-para-mesh.000.vtk')
    meshio_mesh = meshio.read(freecad_mesh_file)
    cell_type = 'tetra'
    points, cells = np.array(meshio_mesh.points), np.array(meshio_mesh.cells_dict[cell_type])
    bundled_info, tet_cells, tet_points = split_tets(cells, points, facet_data.reshape(len(cells), 12, -1), facet_vertices, params)

    print(f"\nprojected area:")
    processed_parea = bundled_info['proj_area'][:12]
    raw_parea = facet_data[:12, 5]
    print(processed_parea)
    print(raw_parea)
    print(processed_parea - raw_parea)

    print(f"\nind_j and ind_j")
    print(bundled_info['ind_i'][:12])
    print(bundled_info['ind_j'][:12])
   
    print(f"\nnormal vectors")
    processed_normal = bundled_info['debug_normal_N'][:12]
    raw_normal = facet_data[:12, 9:12]
    print(processed_normal)
    print(raw_normal)
    print(processed_normal - raw_normal)

    print(f"\nShow all raw data:")
    print(facet_data[:12])

    node_true_ms, node_lumped_ms = compute_mass(len(points), bundled_info, params)

    print(f"\nvolume")
    processed_vol = np.sum(bundled_info['debug_vol'])
    raw_vol = np.sum(node_true_ms[:, 0, 0])
    print(processed_vol)
    print(raw_vol)
    print(processed_vol - raw_vol)


if __name__ == '__main__':
    test()