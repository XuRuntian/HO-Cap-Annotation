"""NIOSH-inspired external measurements on the neutral MANO v1.2 mesh."""

import numpy as np


# Barycentric surface landmarks marked on the projected neutral MANO mesh.
# Skin creases are approximated by anatomical locations, not encoded by MANO.
ANCHORS = {
    "wrist": ((32, 33, 35), (0.085924, 0.315035, 0.599041)),
    "thumb_base": ((124, 125, 123), (0.272693, 0.157914, 0.569393)),
    "index_base": ((168, 171, 138), (0.584185, 0.157789, 0.258026)),
    "middle_base": ((288, 228, 379), (0.048876, 0.065101, 0.886023)),
    "ring_base": ((198, 579, 510), (0.111116, 0.528509, 0.360375)),
    "little_base": ((596, 604, 607), (0.356567, 0.073828, 0.569605)),
    "thumb_tip": ((744,), (1.0,)),
    "index_tip": ((320,), (1.0,)),
    "middle_tip": ((444,), (1.0,)),
    "ring_tip": ((555,), (1.0,)),
    "little_tip": ((672,), (1.0,)),
    "index_edge": ((127, 172), (0.503220, 0.496780)),
    "little_edge": ((610, 611), (0.347289, 0.652711)),
    "palm_index_edge": ((13, 127), (0.617399, 0.382601)),
    "palm_little_edge": ((201, 611), (0.307786, 0.692214)),
}
DIGITS = ("thumb", "index", "middle", "ring", "little")
# Opposite silhouette points at the first knuckle, marked on the neutral projection.
WIDTH_ANCHORS = {
    "thumb": (((702, 709), (0.068554, 0.931446)), ((714, 757), (0.790815, 0.209185))),
    "index": (((58, 134), (0.640606, 0.359394)), ((174, 225), (0.298946, 0.701054))),
    "middle": (((361, 362), (0.156112, 0.843888)), ((381, 394), (0.027112, 0.972888))),
    "ring": (((483, 495), (0.345455, 0.654545)), ((491, 504), (0.050927, 0.949073))),
    "little": (((599, 613), (0.483899, 0.516101)), ((609, 622), (0.419187, 0.580813))),
}
for digit, pair in WIDTH_ANCHORS.items():
    for i, anchor in enumerate(pair):
        ANCHORS[f"{digit}_width_{i}"] = anchor
NIOSH_METRICS = ("hand_length", "hand_breadth", "palm_length", "palm_breadth",
                 *(f"{d}_{measure}" for d in DIGITS for measure in ("length", "breadth")))


def surface_points(vertices):
    """Return projected neutral surface landmarks in model mm, not skeleton joints."""
    vertices = np.asarray(vertices)
    if vertices.shape != (778, 3) or not np.isfinite(vertices).all():
        raise ValueError("Expected a finite [778,3] neutral MANO mesh")
    plane = vertices[:, (0, 2)]
    return {name: np.asarray(w) @ plane[list(ids)] for name, (ids, w) in ANCHORS.items()}


def measure_surface(vertices):
    p = surface_points(vertices)

    def distance(a, b):
        return float(np.linalg.norm(p[a] - p[b]))

    palm_length = distance("wrist", "middle_base")
    digit_lengths = {f"{d}_length": distance(f"{d}_base", f"{d}_tip") for d in DIGITS}
    return {
        # Figure 1 defines overall length as palm length plus middle finger length.
        "hand_length": palm_length + digit_lengths["middle_length"],
        "hand_breadth": distance("index_edge", "little_edge"),
        "palm_length": palm_length,
        "palm_breadth": distance("palm_index_edge", "palm_little_edge"),
        **digit_lengths,
        **{f"{d}_breadth": distance(f"{d}_width_0", f"{d}_width_1") for d in DIGITS},
    }


def neutral_surface_geometry(mano, side):
    """Build a shape-to-mesh function from the native side's flat-pose shape basis."""
    import torch
    from manopth.manolayer import ManoLayer

    layer = ManoLayer(mano_root=str(mano), side=side, use_pca=False,
                      flat_hand_mean=True).double()
    topology = {tuple(sorted(face)) for face in layer.th_faces.tolist()}
    edges = {pair for a, b, c in topology for pair in ((a, b), (a, c), (b, c))}
    if any((len(ids) == 3 and tuple(sorted(ids)) not in topology)
           or (len(ids) == 2 and tuple(sorted(ids)) not in edges)
           for ids, _ in ANCHORS.values()):
        raise ValueError(f"{side}: unsupported MANO mesh topology for surface measurements")
    with torch.no_grad():
        vertices, _ = layer(torch.zeros((11, 48), dtype=torch.float64),
                            torch.cat((torch.zeros((1, 10)), torch.eye(10))).double())
    neutral, *samples = vertices.numpy()
    basis = np.asarray(samples) - neutral

    def predict(beta):
        return neutral + np.einsum("k,kvc->vc", beta, basis)

    return predict
