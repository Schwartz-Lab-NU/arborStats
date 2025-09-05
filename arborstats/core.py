# arborStats/core.py
import numpy as np
from shapely.geometry import Polygon
from concave_hull import concave_hull

def load_swc(path):
    """
    Reads an SWC morphology file and returns:
      • coords:   N×3 numpy array of node positions (X, Y, Z)
      • radii:    length-N numpy array of node radii
      • edges:    list of (node_id, parent_id) tuples for every node including the root with parent -1
    """
    coords = []
    radii = []
    parents = []
    node_ids = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            node_id = int(parts[0])
            x, y, z = map(float, parts[2:5])
            r = float(parts[5])
            parent = int(parts[6])
            node_ids.append(node_id)
            coords.append([x, y, z])
            radii.append(r)
            parents.append(parent)
    coords = np.array(coords, dtype=float)
    radii = np.array(radii, dtype=float)
    # Map original SWC node IDs to array indices
    id_to_idx = {nid: idx for idx, nid in enumerate(node_ids)}
    # Build index-based edges list: (node_index, parent_index) or -1 if no parent
    edges_idx = []
    for idx, parent in enumerate(parents):
        parent_idx = id_to_idx[parent] if parent >= 0 and parent in id_to_idx else -1
        edges_idx.append((idx, parent_idx))
    return coords, radii, edges_idx

def arborStatsFromSkeleton(nodes, edges, radii=None):
    """
    Compute arbor statistics from skeleton nodes and edges.

    Parameters:
        nodes (np.ndarray): N×3 array of node coordinates.
        edges (list of tuple): List of (node_id, parent_id) tuples.

    Returns:
        dict: {
            'num_nodes', 'num_edges', 'total_length',
            'boundary_points', 'polygon_area', 'polygon_area_convex',
            'convexity_index', 'arbor_density',
            'branch_starts', 'branch_ends', 'branch_lens',
            'branch_len_euc', 'branch_angles',
            'branch_tortuosity', 'Nbranches', 'arbor_complexity'
        }
    """
    num_nodes = nodes.shape[0]
    num_edges = len(edges)

    # Project to 2D
    points2d = nodes[:, :2]
    print(f"Number of nodes: {num_nodes}, edges: {num_edges}")

    # compute concave hull quickly using Concaveman
    pts = points2d.tolist()
    hull_pts = concave_hull(pts, concavity=1.0, length_threshold=0)
    print(f"Concave hull points: {len(hull_pts)}")    
    boundary_points = np.array(hull_pts)
    print(f"Boundary points: {boundary_points.shape}")
    # compute area from the polygon
    poly = Polygon(hull_pts)
    polygon_area = poly.area

    # convex hull area and convexity index via polygon convex_hull
    polygon_area_convex = poly.convex_hull.area
    convexity_index = polygon_area_convex / polygon_area if polygon_area > 0 else np.nan

    # total length
    total_length = sum(np.linalg.norm(nodes[n] - nodes[p]) for n, p in edges)
    arbor_density = total_length / polygon_area if polygon_area > 0 else np.nan

    # Compute median of radii if provided
    radii_median = np.median(radii) if radii is not None else np.nan

    # Branch detection
    branch_starts = []
    branch_ends = []
    branch_lens = []
    branch_len_euc = []
    branch_angles = []
    cur_len = 0.0
    curZ = []
    b = -1
    new_branch = True

    print(f"Processing {num_edges} edges for branches")
    for i in range(num_edges):
        n, p = edges[i]
        if new_branch:
            # finish previous branch
            if b >= 0:
                branch_end = nodes[n]
                branch_ends.append(branch_end)
                cur_len += np.linalg.norm(prev_point - branch_end)
                branch_lens.append(cur_len)
                start = branch_starts[b]
                branch_len_euc.append(np.linalg.norm(branch_end - start))
            if i == num_edges - 1:
                break
            # start new branch
            b += 1
            start = nodes[p]
            branch_starts.append(start)
            prev_point = nodes[edges[i][0]]
            cur_len = np.linalg.norm(start - prev_point)
            curZ = [start[2]]
            # branch angle
            try:
                p1 = prev_point; p2 = nodes[p]
                indA = p - 1; indB = p + 1
                pA = nodes[indA]; pB = nodes[indB]
                V1 = p2 - p1; V2 = p2 - pA; V3 = pB - p2
                s1 = np.arctan2(np.linalg.norm(np.cross(V1, V2)), np.dot(V1, V2))
                s2 = np.arctan2(np.linalg.norm(np.cross(V1, V3)), np.dot(V1, V3))
                s3 = np.arctan2(np.linalg.norm(np.cross(V2, V3)), np.dot(V2, V3))
                s1 = abs(min(s1, np.pi - s1))
                s2 = abs(min(s2, np.pi - s2))
                s3 = abs(min(s3, np.pi - s3))
                branch_angles.append(max(s1, s2, s3))
            except:
                branch_angles.append(0.0)
        else:
            p1 = nodes[n]; p2 = nodes[p]
            cur_len += np.linalg.norm(p1 - p2)
            curZ.append(p1[2])
        # determine if next is new branch
        if i >= num_edges - 1:
            new_branch = True
        else:
            nn, pp = edges[i+1]
            new_branch = not ((nn - pp) == 1)

    branch_tortuosity = [l / eu if eu > 0 else np.nan
                         for l, eu in zip(branch_lens, branch_len_euc)]
    Nbranches = len(branch_angles)
    arbor_complexity = Nbranches / total_length if total_length > 0 else np.nan

    # Convert branch angles from radians to degrees
    branch_angles = [ang * (180.0/np.pi) for ang in branch_angles]

    # Compute medians of per-branch statistics
    branch_lens_median = np.median(branch_lens) if branch_lens else np.nan
    branch_angles_median = np.median(branch_angles) if branch_angles else np.nan
    branch_tortuosity_median = np.nanmedian(branch_tortuosity) if branch_tortuosity else np.nan

    stats = {
        'num_edges': num_edges,
        'total_length': total_length,
        'radii_median': radii_median,
        'boundary_points': boundary_points,
        'polygon_area': polygon_area,
        'convexity_index': convexity_index,
        'arbor_density': arbor_density,
        'branch_lens': np.array(branch_lens),
        'branch_angles': np.array(branch_angles),
        'branch_tortuosity': np.array(branch_tortuosity),
        'branch_lens_median': branch_lens_median,
        'branch_angles_median': branch_angles_median,
        'branch_tortuosity_median': branch_tortuosity_median,
        'Nbranches': Nbranches,
        'arbor_complexity': arbor_complexity
    }
    units = {
        'num_edges': 'count',
        'total_length': 'µm',
        'radii_median': 'µm',
        'polygon_area': 'μm^2',
        'convexity_index': None,
        'arbor_density': 'branches/μm^2',
        'branch_lens': 'µm',
        'branch_angles': 'degrees',
        'branch_tortuosity': None,
        'branch_lens_median': 'μm',
        'branch_angles_median': 'degrees',
        'branch_tortuosity_median': None,
        'Nbranches': 'count',
        'arbor_complexity': 'branches/μm'
    }
    print(f"Computed arbor stats: {stats.keys()}")
    # Round statistics: nearest int for total_length & polygon_area; 3 decimal places for others
    for k, v in stats.items():
        if isinstance(v, float):
            if k in ('total_length', 'polygon_area'):
                stats[k] = int(round(v))
            else:
                stats[k] = round(v, 3)
    return stats, units
