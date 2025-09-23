import numpy as np
import torch
import igl
import os
import torch_mesh_ops as TMO
import scipy.sparse as sps


def _torch_batched_coo_slice_to_scipy_csr(x_bcoo: torch.Tensor, batch_index: int, shape):
    if not x_bcoo.is_sparse:
        raise TypeError("Expected a batched torch sparse COO tensor.")
    x = x_bcoo.coalesce()
    idx = x.indices()  # (3, nnz): [batch, row, col]
    vals = x.values()
    mask = (idx[0] == batch_index)
    if mask.sum().item() == 0:
        return sps.csr_matrix(shape)
    rows = idx[1, mask].cpu().numpy()
    cols = idx[2, mask].cpu().numpy()
    data = vals[mask].detach().cpu().numpy()
    csr = sps.coo_matrix((data, (rows, cols)), shape=shape).tocsr()
    csr.sum_duplicates()
    csr.eliminate_zeros()
    return csr


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
    if dtype == torch.float64:
        return {
            'lap_rel': 1e-9,
            'energy_rel': 1e-10,
            'areas_rel': 1e-10,
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


def _load_batched_meshes(dir_path):
    files = sorted([f for f in os.listdir(dir_path) if f.endswith('.obj')])
    paths = [os.path.join(dir_path, f) for f in files]
    meshes = [load_mesh(p) for p in paths]
    return paths, meshes


def validate_cotangent_laplacian_batched(meshes_dir, precision=torch.float32):
    paths, meshes = _load_batched_meshes(meshes_dir)
    B = len(meshes)

    verts_list = [torch.from_numpy(v).to(precision).to('cuda') for (v, _) in meshes]
    faces_list = [torch.from_numpy(f).long().to('cuda') for (_, f) in meshes]

    # Build batched tensors
    # Faces may differ per-batch; stack as (B,F,3)
    V = torch.stack(verts_list, dim=0)  # (B, nV, 3)
    F = torch.stack(faces_list, dim=0)  # (B, nF, 3)

    L_b = TMO.cotangent_laplacian_batched(V, F)  # (B, nV, nV) sparse

    th = _thresholds_for_precision(precision)['lap_rel']
    for b in range(B):
        verts_np, faces_np = meshes[b]
        L = _torch_batched_coo_slice_to_scipy_csr(L_b, b, (verts_np.shape[0], verts_np.shape[0]))
        L_igl = (-igl.cotmatrix(verts_np, faces_np)).tocsr()
        L_igl.sum_duplicates(); L_igl.eliminate_zeros()

        rel = _sparse_rel_fro(L, L_igl)
        diff = (L - L_igl).tocsr(); diff.sum_duplicates(); diff.eliminate_zeros()
        max_abs = float(np.max(np.abs(diff.data))) if diff.nnz > 0 else 0.0
        ok = rel <= th
        print(f"cotangent_laplacian_batched | {os.path.basename(paths[b])} | {_status(ok)} rel_fro={rel:.3e} (tol={th:.1e}), max_abs={max_abs:.3e}, nnz_diff={diff.nnz}")
        if not ok:
            print("WARNING: Batched cotangent Laplacian relative error exceeds tolerance.")


def validate_face_areas_batched(meshes_dir, precision=torch.float32):
    paths, meshes = _load_batched_meshes(meshes_dir)
    B = len(meshes)

    verts_list = [torch.from_numpy(v).to(precision).to('cuda') for (v, _) in meshes]
    faces_list = [torch.from_numpy(f).long().to('cuda') for (_, f) in meshes]
    V = torch.stack(verts_list, dim=0)
    F = torch.stack(faces_list, dim=0)

    areas_b = TMO.face_areas_batched(V, F).cpu().numpy()  # (B, F)

    th = _thresholds_for_precision(precision)['areas_rel']
    for b in range(B):
        verts_np, faces_np = meshes[b]
        areas_igl = igl.doublearea(verts_np, faces_np) / 2.0
        areas = areas_b[b]
        error = np.abs(areas - areas_igl)
        rel_l2 = float(np.linalg.norm(error) / max(np.linalg.norm(areas_igl), 1e-16))
        max_rel = float(np.max(error / np.maximum(np.abs(areas_igl), 1e-16))) if areas_igl.size else 0.0
        ok = rel_l2 <= th
        print(f"face_areas_batched          | {os.path.basename(paths[b])} | {_status(ok)} rel_l2={rel_l2:.3e} (tol={th:.1e}), max_rel={max_rel:.3e}, max_abs={error.max():.3e}")
        if not ok:
            print("WARNING: Batched face areas relative error exceeds tolerance.")


def validate_intrinsic_gradient_batched(meshes_dir, precision=torch.float32):
    paths, meshes = _load_batched_meshes(meshes_dir)
    B = len(meshes)

    verts_list = [torch.from_numpy(v).to(precision).to('cuda') for (v, _) in meshes]
    faces_list = [torch.from_numpy(f).long().to('cuda') for (_, f) in meshes]

    V = torch.stack(verts_list, dim=0)  # (B, V, 3)
    F = torch.stack(faces_list, dim=0)  # (B, F, 3)

    edge_lens = TMO.edge_lengths_batched(V, F)  # (B, F, 3)
    G_b = TMO.intrinsic_gradient_batched(edge_lens, F)  # (B, 2F, V) sparse

    th_lap = _thresholds_for_precision(precision)['lap_rel']
    th_energy = _thresholds_for_precision(precision)['energy_rel']
    rng = np.random.default_rng(0)

    for b in range(B):
        verts_np, faces_np = meshes[b]
        nV = verts_np.shape[0]
        nF = faces_np.shape[0]

        G = _torch_batched_coo_slice_to_scipy_csr(G_b, b, (2 * nF, nV))
        areas = igl.doublearea(verts_np, faces_np) / 2.0
        areas_interleaved = np.repeat(areas, 2)

        A = sps.diags(areas_interleaved)
        L_from_G = (G.T @ (A @ G)).tocsr()
        L_from_G.sum_duplicates(); L_from_G.eliminate_zeros()

        L_igl = (-igl.cotmatrix(verts_np, faces_np)).tocsr()
        L_igl.sum_duplicates(); L_igl.eliminate_zeros()

        rel = _sparse_rel_fro(L_from_G, L_igl)
        diff = (L_from_G - L_igl).tocsr(); diff.sum_duplicates(); diff.eliminate_zeros()
        max_abs = float(np.max(np.abs(diff.data))) if diff.nnz > 0 else 0.0
        ok_lap = rel <= th_lap
        print(f"intrinsic_gradient_batched  | {os.path.basename(paths[b])} | {_status(ok_lap)} L_from_G vs L_igl: rel_fro={rel:.3e} (tol={th_lap:.1e}), max_abs={max_abs:.3e}, nnz_diff={diff.nnz}")
        if not ok_lap:
            print("WARNING: Batched G-derived Laplacian mismatch above tolerance.")

        for t in range(5):
            u = rng.standard_normal(nV).astype(areas.dtype)
            gu = (G @ u).reshape(-1, 2)
            E_G = float((areas * (gu**2).sum(axis=1)).sum())
            E_L = float(u @ (L_igl @ u))
            rel_e = abs(E_G - E_L) / (abs(E_L) + 1e-16)
            ok_e = rel_e <= th_energy
            print(f"  energy check {t}           | {_status(ok_e)} rel_err={rel_e:.3e} (tol={th_energy:.1e}); E_G={E_G:.6e}, E_L={E_L:.6e}")
            if not ok_e:
                print("  WARNING: Batched Dirichlet energy discrepancy above tolerance.")


def validate_intrinsic_gradient_stacked_batched(meshes_dir, precision=torch.float32):
    paths, meshes = _load_batched_meshes(meshes_dir)
    B = len(meshes)

    verts_list = [torch.from_numpy(v).to(precision).to('cuda') for (v, _) in meshes]
    faces_list = [torch.from_numpy(f).long().to('cuda') for (_, f) in meshes]

    V = torch.stack(verts_list, dim=0)
    F = torch.stack(faces_list, dim=0)

    edge_lens = TMO.edge_lengths_batched(V, F)  # (B, F, 3)
    G_b = TMO.intrinsic_gradient_stacked_batched(edge_lens, F)  # (B, 2F, V) sparse

    th_lap = _thresholds_for_precision(precision)['lap_rel']
    th_energy = _thresholds_for_precision(precision)['energy_rel']
    rng = np.random.default_rng(0)

    for b in range(B):
        verts_np, faces_np = meshes[b]
        nV = verts_np.shape[0]
        nF = faces_np.shape[0]

        G = _torch_batched_coo_slice_to_scipy_csr(G_b, b, (2 * nF, nV))
        areas = igl.doublearea(verts_np, faces_np) / 2.0
        areas_stacked = np.tile(areas, 2)

        A = sps.diags(areas_stacked)
        L_from_G = (G.T @ (A @ G)).tocsr()
        L_from_G.sum_duplicates(); L_from_G.eliminate_zeros()

        L_igl = (-igl.cotmatrix(verts_np, faces_np)).tocsr()
        L_igl.sum_duplicates(); L_igl.eliminate_zeros()

        rel = _sparse_rel_fro(L_from_G, L_igl)
        diff = (L_from_G - L_igl).tocsr(); diff.sum_duplicates(); diff.eliminate_zeros()
        max_abs = float(np.max(np.abs(diff.data))) if diff.nnz > 0 else 0.0
        ok_lap = rel <= th_lap
        print(f"intrinsic_grad_stack_batched | {os.path.basename(paths[b])} | {_status(ok_lap)} L_from_G vs L_igl: rel_fro={rel:.3e} (tol={th_lap:.1e}), max_abs={max_abs:.3e}, nnz_diff={diff.nnz}")
        if not ok_lap:
            print("WARNING: Batched stacked G-derived Laplacian mismatch above tolerance.")

        for t in range(5):
            u = rng.standard_normal(nV).astype(areas.dtype)
            gu = (G @ u)
            gu = np.column_stack((gu[:nF], gu[nF:]))
            E_G = float((areas * (gu**2).sum(axis=1)).sum())
            E_L = float(u @ (L_igl @ u))
            rel_e = abs(E_G - E_L) / (abs(E_L) + 1e-16)
            ok_e = rel_e <= th_energy
            print(f"  energy check {t}           | {_status(ok_e)} rel_err={rel_e:.3e} (tol={th_energy:.1e}); E_G={E_G:.6e}, E_L={E_L:.6e}")
            if not ok_e:
                print("  WARNING: Batched Dirichlet energy discrepancy above tolerance.")


def validate_vertex_mass_batched(meshes_dir, precision=torch.float32):
    paths, meshes = _load_batched_meshes(meshes_dir)
    B = len(meshes)

    verts_list = [torch.from_numpy(v).to(precision).to('cuda') for (v, _) in meshes]
    faces_list = [torch.from_numpy(f).long().to('cuda') for (_, f) in meshes]
    V = torch.stack(verts_list, dim=0)
    F = torch.stack(faces_list, dim=0)

    masses = TMO.vertex_mass_batched(V, F).cpu().numpy()  # (B, nV)
    th = _thresholds_for_precision(precision)['areas_rel']

    for b in range(B):
        verts_np, faces_np = meshes[b]
        mass_igl = igl.massmatrix(verts_np, faces_np, igl.MASSMATRIX_TYPE_BARYCENTRIC).diagonal()
        mass = masses[b]
        error = np.abs(mass - mass_igl)
        rel_l2 = float(np.linalg.norm(error) / max(np.linalg.norm(mass_igl), 1e-16))
        max_rel = float(np.max(error / np.maximum(np.abs(mass_igl), 1e-16))) if mass_igl.size else 0.0
        ok = rel_l2 <= th
        print(f"vertex_mass_batched        | {os.path.basename(paths[b])} | {_status(ok)} rel_l2={rel_l2:.3e} (tol={th:.1e}), max_rel={max_rel:.3e}, max_abs={error.max():.3e}")
        if not ok:
            print("WARNING: Batched vertex mass relative error exceeds tolerance.")


def validate_edge_lengths_batched(meshes_dir, precision=torch.float32):
    paths, meshes = _load_batched_meshes(meshes_dir)
    B = len(meshes)

    verts_list = [torch.from_numpy(v).to(precision).to('cuda') for (v, _) in meshes]
    faces_list = [torch.from_numpy(f).long().to('cuda') for (_, f) in meshes]
    V = torch.stack(verts_list, dim=0)
    F = torch.stack(faces_list, dim=0)

    edge_lens = TMO.edge_lengths_batched(V, F).cpu().numpy()  # (B, F, 3)
    th = _thresholds_for_precision(precision)['areas_rel']

    for b in range(B):
        verts_np, faces_np = meshes[b]
        edge_igl = igl.edge_lengths(verts_np, faces_np)
        error = np.abs(edge_lens[b] - edge_igl)
        rel_l2 = float(np.linalg.norm(error) / max(np.linalg.norm(edge_igl), 1e-16))
        max_rel = float(np.max(error / np.maximum(np.abs(edge_igl), 1e-16))) if edge_igl.size else 0.0
        ok = rel_l2 <= th
        print(f"edge_lengths_batched       | {os.path.basename(paths[b])} | {_status(ok)} rel_l2={rel_l2:.3e} (tol={th:.1e}), max_rel={max_rel:.3e}, max_abs={error.max():.3e}")
        if not ok:
            print("WARNING: Batched edge lengths relative error exceeds tolerance.")


if __name__ == "__main__":
    here = os.path.dirname(__file__)

    # batches_meshes/ contains meshes with the exact same topology
    # meshes_dir = os.path.join(here, "batched_meshes")

    # inhomogeneous_meshes/ contains meshes with different connectivity, but same num. verts/faces:
    meshes_dir = os.path.join(here, "inhomogeneous_meshes")
    funcs = [
        validate_cotangent_laplacian_batched,
        validate_face_areas_batched,
        validate_intrinsic_gradient_batched,
        validate_vertex_mass_batched,
        validate_edge_lengths_batched,
        validate_intrinsic_gradient_stacked_batched,
    ]

    precision = torch.float64
    print(f" ----- Batched testing with precision: {precision} -----")
    for func in funcs:
        func(meshes_dir, precision=precision)

    precision = torch.float32
    print()
    print(f" ----- Batched testing with precision: {precision} -----")
    for func in funcs:
        func(meshes_dir, precision=precision)
