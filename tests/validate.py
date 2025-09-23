import numpy as np
import torch
import igl
import os
import torch_mesh_ops as TMO
import scipy.sparse as sps

def _torch_coo_to_scipy_csr(x_coo: torch.Tensor) -> sps.csr_matrix:
    if not x_coo.is_sparse:
        raise TypeError("Expected a torch sparse COO tensor.")
    x = x_coo.coalesce()  # sum duplicates, sort indices
    rows, cols = x.indices()
    vals = x.values()
    rows = rows.cpu().numpy()
    cols = cols.cpu().numpy()
    vals = vals.detach().cpu().numpy()
    csr = sps.coo_matrix((vals, (rows, cols)), shape=x.shape).tocsr()
    csr.sum_duplicates()
    csr.eliminate_zeros()
    return csr

def _thresholds_for_precision(dtype: torch.dtype):
    """Choose tolerances based on dtype."""
    if dtype == torch.float64:
        return {
            'lap_rel': 1e-9,       # relative Frobenius norm for Laplacians
            'energy_rel': 1e-10,   # relative Dirichlet energy error
            'areas_rel': 1e-10,    # relative L2 error for areas
        }
    return {
        'lap_rel': 5e-5,
        'energy_rel': 5e-5,
        'areas_rel': 5e-5,
    }

def _status(ok: bool) -> str:
    return "[OK]" if ok else "[FAIL]"

def _sparse_rel_fro(A: sps.spmatrix, B: sps.spmatrix) -> float:
    diff = (A - B).tocsr()
    diff.sum_duplicates(); diff.eliminate_zeros()
    num = sps.linalg.norm(diff)
    den = sps.linalg.norm(B)
    den = max(den, 1e-16)
    return float(num / den)

def load_mesh(path):
    verts, faces = igl.read_triangle_mesh(path)
    return verts, faces

def validate_cotangent_laplacian(mesh_path, precision=torch.float32):
    verts_np, faces_np = load_mesh(mesh_path)

    verts = torch.from_numpy(verts_np).to(precision).to('cuda')
    faces = torch.from_numpy(faces_np).long().to('cuda')
    
    L = TMO.cotangent_laplacian(verts, faces)
    L = _torch_coo_to_scipy_csr(L)

    L_igl = -igl.cotmatrix(verts_np, faces_np) # scipy csc_matrix
    L_igl = L_igl.tocsr()
    L_igl.sum_duplicates()
    L_igl.eliminate_zeros()

    rel = _sparse_rel_fro(L, L_igl)
    diff = (L - L_igl).tocsr(); diff.sum_duplicates(); diff.eliminate_zeros()
    max_abs = float(np.max(np.abs(diff.data))) if diff.nnz > 0 else 0.0
    th = _thresholds_for_precision(precision)['lap_rel']
    ok = rel <= th
    print(f"cotangent_laplacian | {os.path.basename(mesh_path)} | { _status(ok) } rel_fro={rel:.3e} (tol={th:.1e}), max_abs={max_abs:.3e}, nnz_diff={diff.nnz}")
    if not ok:
        print("WARNING: Cotangent Laplacian relative error exceeds tolerance.")

def validate_face_areas(mesh_path, precision=torch.float32):
    verts_np, faces_np = load_mesh(mesh_path)
    
    verts = torch.from_numpy(verts_np).to(precision).to('cuda')
    faces = torch.from_numpy(faces_np).long().to('cuda')

    areas = TMO.face_areas(verts, faces).cpu().numpy()
    areas_igl = igl.doublearea(verts_np, faces_np) / 2.0

    error = np.abs(areas - areas_igl)
    rel_l2 = float(np.linalg.norm(error) / max(np.linalg.norm(areas_igl), 1e-16))
    max_rel = float(np.max(error / np.maximum(np.abs(areas_igl), 1e-16))) if areas_igl.size else 0.0
    th = _thresholds_for_precision(precision)['areas_rel']
    ok = rel_l2 <= th
    print(f"face_areas          | {os.path.basename(mesh_path)} | { _status(ok) } rel_l2={rel_l2:.3e} (tol={th:.1e}), max_rel={max_rel:.3e}, max_abs={error.max():.3e}")
    if not ok:
        print("WARNING: Face areas relative error exceeds tolerance.")

def validate_intrinsic_gradient_stacked(mesh_path, precision=torch.float32):
    """
    It's difficult to compare two gradient operators directly, due to 
    different choice of tangent basis and element ordering. So this 
    function does two things:

    1) compares Laplacian assembled from our gradient operator to igl.cotmatrix:
        L = G^T A G, where A is a diagonal matrix of face areas

    2) compares Dirichlet energies computed (on random signals) using our gradient 
        operator and igl.cotmatrix
    """
    verts_np, faces_np = load_mesh(mesh_path)

    # Our gradient operator
    verts = torch.from_numpy(verts_np).to(precision).to('cuda')
    faces = torch.from_numpy(faces_np).long().to('cuda')
    edge_lens = TMO.edge_lengths(verts, faces)

    G = TMO.intrinsic_gradient_stacked(edge_lens, faces)
    G = _torch_coo_to_scipy_csr(G)  # (2F, V)

    areas = igl.doublearea(verts_np, faces_np) / 2.0  # (F,)
    areas_stacked = np.tile(areas, 2)

    # Assemble Laplacian from G
    A = sps.diags(areas_stacked)
    L_from_G = (G.T @ (A @ G)).tocsr()
    L_from_G.sum_duplicates(); L_from_G.eliminate_zeros()

    L_igl = (-igl.cotmatrix(verts_np, faces_np)).tocsr()
    L_igl.sum_duplicates(); L_igl.eliminate_zeros()

    # Relative Frobenius difference between Laplacians
    rel = _sparse_rel_fro(L_from_G, L_igl)
    th_lap = _thresholds_for_precision(precision)['lap_rel']
    ok_lap = rel <= th_lap
    diff = (L_from_G - L_igl).tocsr(); diff.sum_duplicates(); diff.eliminate_zeros()
    max_abs = float(np.max(np.abs(diff.data))) if diff.nnz > 0 else 0.0
    print(f"intrinsic_gradient  | {os.path.basename(mesh_path)} | {_status(ok_lap)} L_from_G vs L_igl: rel_fro={rel:.3e} (tol={th_lap:.1e}), max_abs={max_abs:.3e}, nnz_diff={diff.nnz}")
    if not ok_lap:
        print("WARNING: L_from_G vs igl cotmatrix mismatch above tolerance.")

    # Random-signal Dirichlet energy checks (basis-invariant)
    nV = verts_np.shape[0]
    nF = faces_np.shape[0]
    rng = np.random.default_rng(0)
    th_energy = _thresholds_for_precision(precision)['energy_rel']
    for t in range(5):
        u = rng.standard_normal(nV).astype(areas.dtype)
        # Energy via gradient: sum_f area_f * ||(G u)_f||^2
        gu = (G @ u)
        gu = np.column_stack((gu[:nF], gu[nF:]))
        E_G = float((areas * (gu**2).sum(axis=1)).sum())
        # Energy via Laplacian (igl)
        E_L = float(u @ (L_igl @ u))
        rel_e = abs(E_G - E_L) / (abs(E_L) + 1e-16)
        ok_e = rel_e <= th_energy
        print(f"  energy check {t}    | {_status(ok_e)} rel_err={rel_e:.3e} (tol={th_energy:.1e}); E_G={E_G:.6e}, E_L={E_L:.6e}")
        if not ok_e:
            print("  WARNING: Dirichlet energy discrepancy above tolerance.")

def validate_intrinsic_gradient(mesh_path, precision=torch.float32):
    verts_np, faces_np = load_mesh(mesh_path)

    # Our gradient operator
    verts = torch.from_numpy(verts_np).to(precision).to('cuda')
    faces = torch.from_numpy(faces_np).long().to('cuda')
    edge_lens = TMO.edge_lengths(verts, faces)

    G = TMO.intrinsic_gradient(edge_lens, faces)
    G = _torch_coo_to_scipy_csr(G)  # (2F, V)

    areas = igl.doublearea(verts_np, faces_np) / 2.0  # (F,)
    areas_interleaved = np.repeat(areas, 2)

    # Assemble Laplacian from G
    A = sps.diags(areas_interleaved)
    L_from_G = (G.T @ (A @ G)).tocsr()
    L_from_G.sum_duplicates(); L_from_G.eliminate_zeros()

    L_igl = (-igl.cotmatrix(verts_np, faces_np)).tocsr()
    L_igl.sum_duplicates(); L_igl.eliminate_zeros()

    # Relative Frobenius difference between Laplacians
    rel = _sparse_rel_fro(L_from_G, L_igl)
    th_lap = _thresholds_for_precision(precision)['lap_rel']
    ok_lap = rel <= th_lap
    diff = (L_from_G - L_igl).tocsr(); diff.sum_duplicates(); diff.eliminate_zeros()
    max_abs = float(np.max(np.abs(diff.data))) if diff.nnz > 0 else 0.0
    print(f"intrinsic_gradient  | {os.path.basename(mesh_path)} | {_status(ok_lap)} L_from_G vs L_igl: rel_fro={rel:.3e} (tol={th_lap:.1e}), max_abs={max_abs:.3e}, nnz_diff={diff.nnz}")
    if not ok_lap:
        print("WARNING: L_from_G vs igl cotmatrix mismatch above tolerance.")

    # Random-signal Dirichlet energy checks (basis-invariant)
    nV = verts_np.shape[0]
    rng = np.random.default_rng(0)
    th_energy = _thresholds_for_precision(precision)['energy_rel']
    for t in range(5):
        u = rng.standard_normal(nV).astype(areas.dtype)
        # Energy via gradient: sum_f area_f * ||(G u)_f||^2
        gu = (G @ u).reshape(-1, 2)
        E_G = float((areas * (gu**2).sum(axis=1)).sum())
        # Energy via Laplacian (igl)
        E_L = float(u @ (L_igl @ u))
        rel_e = abs(E_G - E_L) / (abs(E_L) + 1e-16)
        ok_e = rel_e <= th_energy
        print(f"  energy check {t}    | {_status(ok_e)} rel_err={rel_e:.3e} (tol={th_energy:.1e}); E_G={E_G:.6e}, E_L={E_L:.6e}")
        if not ok_e:
            print("  WARNING: Dirichlet energy discrepancy above tolerance.")

def validate_vertex_mass(mesh_path, precision=torch.float32):
    verts_np, faces_np = load_mesh(mesh_path)
    verts = torch.from_numpy(verts_np).to(precision).to('cuda')
    faces = torch.from_numpy(faces_np).long().to('cuda')

    mass = TMO.vertex_mass(verts, faces).cpu().numpy()
    mass_igl = igl.massmatrix(verts_np, faces_np, igl.MASSMATRIX_TYPE_BARYCENTRIC)
    mass_igl = mass_igl.diagonal()
    
    error = np.abs(mass - mass_igl)
    rel_l2 = float(np.linalg.norm(error) / max(np.linalg.norm(mass_igl), 1e-16))
    max_rel = float(np.max(error / np.maximum(np.abs(mass_igl), 1e-16))) if mass_igl.size else 0.0
    th = _thresholds_for_precision(precision)['areas_rel']
    ok = rel_l2 <= th
    print(f"vertex_mass         | {os.path.basename(mesh_path)} | { _status(ok) } rel_l2={rel_l2:.3e} (tol={th:.1e}), max_rel={max_rel:.3e}, max_abs={error.max():.3e}")
    if not ok:
        print("WARNING: Vertex mass relative error exceeds tolerance.")

def validate_edge_lengths(mesh_path, precision=torch.float32):
    verts_np, faces_np = load_mesh(mesh_path)
    verts = torch.from_numpy(verts_np).to(precision).to('cuda')
    faces = torch.from_numpy(faces_np).long().to('cuda')

    edge_lengths = TMO.edge_lengths(verts, faces).cpu().numpy()
    edge_lengths_igl = igl.edge_lengths(verts_np, faces_np)

    error = np.abs(edge_lengths - edge_lengths_igl)
    rel_l2 = float(np.linalg.norm(error) / max(np.linalg.norm(edge_lengths_igl), 1e-16))
    max_rel = float(np.max(error / np.maximum(np.abs(edge_lengths_igl), 1e-16))) if edge_lengths_igl.size else 0.0
    th = _thresholds_for_precision(precision)['areas_rel']
    ok = rel_l2 <= th
    print(f"edge_lengths        | {os.path.basename(mesh_path)} | { _status(ok) } rel_l2={rel_l2:.3e} (tol={th:.1e}), max_rel={max_rel:.3e}, max_abs={error.max():.3e}")
    if not ok:
        print("WARNING: Edge lengths relative error exceeds tolerance.")

if __name__ == "__main__":
    all_meshes = [os.path.join("./meshes", f) for f in os.listdir("./meshes") if f.endswith(".obj")]
    funcs = [validate_cotangent_laplacian, validate_face_areas, validate_intrinsic_gradient, validate_vertex_mass, validate_edge_lengths, validate_intrinsic_gradient_stacked]

    precision = torch.float64
    print(f" ----- Testing with precision: {precision} -----")
    for func in funcs:
        for mesh in all_meshes:
            func(mesh, precision=precision)
    
    precision = torch.float32
    print()
    print(f" ----- Testing with precision: {precision} -----")
    for func in funcs:
        for mesh in all_meshes:
            func(mesh, precision=precision)
