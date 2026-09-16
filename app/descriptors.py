import io
import base64
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import Draw, rdFingerprintGenerator
from sklearn.manifold import TSNE

def smiles_to_ecfp4(smiles: str, n_bits: int = 2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits, dtype=np.uint8)
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=n_bits)
    fp = generator.GetFingerprint(mol)
    arr = np.zeros((n_bits,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def compute_similarity_matrix(bitvectors: np.ndarray | list, method: str = "tanimoto"):
    if method not in {"tanimoto", "tversky"}:
        raise ValueError(f"Unsupported similarity method: {method}")

    X = np.asarray(bitvectors, dtype=float)
    intersection = np.dot(X, X.T)
    cardinality = X.sum(axis=1, keepdims=True)

    if method == "tanimoto":
        denominator = cardinality + cardinality.T - intersection
        np.maximum(denominator, 1e-9, out=denominator)
        return intersection / denominator

    # Match the project convention: Tv(A, B) uses alpha=1, beta=0,
    # and the symmetric matrix keeps the larger direction for each pair.
    only_a = cardinality - intersection
    only_b = cardinality.T - intersection
    tversky_ab = intersection / np.maximum(intersection + only_a, 1e-9)
    tversky_ba = intersection / np.maximum(intersection + only_b, 1e-9)
    return np.maximum(tversky_ab, tversky_ba)

def get_mol_b64_image(smiles: str, size=(180, 180)) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    img = Draw.MolToImage(mol, size=size)
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"

def compute_tsne(fingerprints: np.ndarray, perplexity: int = 30, random_state: int = 42):
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state, init="pca", learning_rate="auto")
    return tsne.fit_transform(fingerprints)