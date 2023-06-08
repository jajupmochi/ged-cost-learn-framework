"""
embed



@Author: linlin
@Date: 19.05.23
"""
import numpy as np


# %% ----- the target space: -----

def compute_D_y(
        G_app, y_app, G_test, y_test,
        y_distance=None,
        mode='reg', unlabeled=False, ed_method='bipartite',
        descriptor='atom_bond_types',
        **kwargs
):
    """
    Return the distance matrix directly computed by y.
    """
    N = len(y_app)

    dis_mat = np.zeros((N, N))

    for i in range(N):
        for j in range(i + 1, N):
            dis_mat[i, j] = y_distance(y_app[i], y_app[j])
            dis_mat[j, i] = dis_mat[i, j]

    return dis_mat


# %% -----


def compute_D_embed(
        G_app, y_app, G_test, y_test,
        y_distance=None,
        mode='reg', unlabeled=False, ed_method='bipartite',
        descriptor='atom_bond_types',
        embedding_space='y',
        fit_test=False,
        **kwargs
):
    """
    Evaluate the distance matrix between elements in the embedded space.
    """
    if embedding_space == 'y':
        # Compute distances between elements in embedded space:
        y_dis_mat = compute_D_y(
            G_app, y_app, G_test, y_test,
            y_distance=y_distance,
            mode=mode, unlabeled=unlabeled, ed_method=ed_method,
            descriptor=descriptor,
            fit_test=fit_test,
        )
        return y_dis_mat
    # ---- kernel spaces: -----
    elif embedding_space == 'sp_kernel':
        from .kernel import compute_D_shortest_path_kernel
        return compute_D_shortest_path_kernel(
            G_app, y_app, G_test, y_test,
            y_distance=y_distance,
            mode=mode, unlabeled=unlabeled, ed_method=ed_method,
            descriptor=descriptor,
            fit_test=fit_test,
            **kwargs
        )
    elif embedding_space == 'structural_sp':
        from .kernel import compute_D_structural_sp_kernel
        return compute_D_structural_sp_kernel(
            G_app, y_app, G_test, y_test,
            y_distance=y_distance,
            mode=mode, unlabeled=unlabeled, ed_method=ed_method,
            descriptor=descriptor,
            fit_test=fit_test,
            **kwargs
        )
    elif embedding_space == 'path_kernel':
        from .kernel import compute_D_path_kernel
        return compute_D_path_kernel(
            G_app, y_app, G_test, y_test,
            y_distance=y_distance,
            mode=mode, unlabeled=unlabeled, ed_method=ed_method,
            descriptor=descriptor,
            fit_test=fit_test,
            **kwargs
        )
    elif embedding_space == 'treelet_kernel':
        from .kernel import compute_D_treelet_kernel
        return compute_D_treelet_kernel(
            G_app, y_app, G_test, y_test,
            y_distance=y_distance,
            mode=mode, unlabeled=unlabeled, ed_method=ed_method,
            descriptor=descriptor,
            fit_test=fit_test,
            **kwargs
        )
    elif embedding_space == 'wlsubtree_kernel':
        from .kernel import compute_D_wlsubtree_kernel
        return compute_D_wlsubtree_kernel(
            G_app, y_app, G_test, y_test,
            y_distance=y_distance,
            mode=mode, unlabeled=unlabeled, ed_method=ed_method,
            descriptor=descriptor,
            fit_test=fit_test,
            **kwargs
        )
    # ---- GNN spaces: -----
    elif embedding_space == 'gcn':
        from .gnn import compute_D_gcn
        return compute_D_gcn(
            G_app, y_app, G_test, y_test,
            y_distance=y_distance,
            mode=mode, unlabeled=unlabeled, ed_method=ed_method,
            descriptor=descriptor,
            fit_test=fit_test,
            model_name=embedding_space,
            **kwargs
        )
    elif embedding_space == 'gat':
        from .gnn import compute_D_gat
        return compute_D_gat(
            G_app, y_app, G_test, y_test,
            y_distance=y_distance,
            mode=mode, unlabeled=unlabeled, ed_method=ed_method,
            descriptor=descriptor,
            fit_test=fit_test,
            model_name=embedding_space,
            **kwargs
        )
    else:
        raise ValueError(
            'Unknown embedding space: {}'.format(embedding_space)
        )

