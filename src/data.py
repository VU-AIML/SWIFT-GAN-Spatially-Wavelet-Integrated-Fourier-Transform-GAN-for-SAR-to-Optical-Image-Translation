import os
import glob
import torch
from torch.utils.data import Dataset
import rasterio
from rasterio.windows import Window, from_bounds
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from PIL import Image


class FolderDataset(Dataset):
    """
    Simple single-domain RGB image folder dataset for standard unpaired I2I
    benchmarks (horse2zebra, maps, summer2winter, ...).

    Reads all images from one folder (e.g. trainA) and returns tensors in the
    SAME format as SentinelDataset so the rest of the pipeline (model, losses,
    metrics) is unchanged:
        - shape [3, patch_size, patch_size]
        - float32
        - normalized to [-1, 1]

    Usage (one instance per domain, exactly like the S1/S2 split):
        ds_A = FolderDataset(".../horse2zebra/trainA", patch_size=256, transform=t)
        ds_B = FolderDataset(".../horse2zebra/trainB", patch_size=256, transform=t)
    """
    IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff')

    def __init__(self, root_dir, patch_size=256, transform=None):
        self.root_dir = root_dir
        self.patch_size = patch_size
        self.transform = transform

        # Recursively collect image paths (benchmarks are usually flat, but be safe)
        self.files = []
        for ext in self.IMG_EXTS:
            self.files += glob.glob(os.path.join(root_dir, f"*{ext}"))
            self.files += glob.glob(os.path.join(root_dir, "**", f"*{ext}"), recursive=True)
        # de-dup and sort for determinism
        self.files = sorted(set(self.files))

        if len(self.files) == 0:
            print(f"[WARNING] FolderDataset found 0 images in {root_dir}")
        else:
            print(f"[INFO] FolderDataset: {len(self.files)} images in {root_dir}")

        # A resize to patch_size is always applied so odd-sized benchmark images
        # (e.g. maps are 600x600, some are 256x256) become a fixed square.
        self._resize = transforms.Resize((patch_size, patch_size))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        try:
            img = Image.open(path).convert('RGB')          # force 3-channel
            img = self._resize(img)                          # -> patch_size square
            arr = np.asarray(img, dtype=np.float32) / 255.0  # [0,1], HWC
            arr = np.transpose(arr, (2, 0, 1))               # -> CHW
            img_tensor = torch.from_numpy(arr).float()

            # Same augmentation hook as SentinelDataset (tensor-level transforms:
            # RandomHorizontalFlip / RandomVerticalFlip operate on tensors).
            if self.transform:
                img_tensor = self.transform(img_tensor)

            # Same [-1, 1] normalization as SentinelDataset (CycleGAN convention).
            img_tensor = (img_tensor - 0.5) / 0.5
            return img_tensor

        except Exception as e:
            print(f"[ERROR] FolderDataset failed on {path}: {e}")
            return torch.zeros((3, self.patch_size, self.patch_size))


class SentinelDataset(Dataset):
    def __init__(self, root_dir, domain='s1', patch_size=256, cloud_threshold=0.10, transform=None, cache_index=True):
        """
        Custom Dataset for reading Sentinel-1 (3-Channel) and Sentinel-2 data recursively.
        
        Args:
            root_dir (str): Root path to data (can contain multiple subdirectories).
            domain (str): 's1' for SAR, 's2' for Optical.
            patch_size (int): Size of the image patches.
            cloud_threshold (float): Maximum allowed cloud coverage percentage (for S2).
            transform (callable, optional): PyTorch transforms for augmentation.
            cache_index (bool): Whether to save/load valid patch locations.
        """
        self.root_dir = root_dir
        self.domain = domain
        self.patch_size = patch_size
        self.cloud_threshold = cloud_threshold
        self.transform = transform
        self.valid_patches = [] 

        # Cache file version updated to v8 to force a fresh scan across all subdirectories
        index_file = os.path.join(root_dir, f"{domain}_patch_index_v8.npy")

        if cache_index and os.path.exists(index_file):
            print(f"[INFO] Loading cached index for {domain.upper()} from {index_file}...")
            self.valid_patches = np.load(index_file, allow_pickle=True).tolist()
        else:
            print(f"[INFO] Scanning {domain.upper()} files recursively to build patch index...")
            self._scan_and_index()
            if cache_index and len(self.valid_patches) > 0:
                np.save(index_file, self.valid_patches)
                print(f"[INFO] Index saved to {index_file}")

    def _scan_and_index(self):
        """
        Recursively scans large TIFF files in all subdirectories. 
        Filters out empty SAR patches and cloudy Optical patches.
        """
        stride = self.patch_size
        
        # ---------------------------
        # SENTINEL-1 (SAR) SCANNING
        # ---------------------------
        if self.domain == 's1':
            # Use ** and recursive=True to find all VV files in all subfolders
            search_pattern = os.path.join(self.root_dir, "**", "*VV*.tif")
            vv_files = glob.glob(search_pattern, recursive=True)
            
            for vv_path in tqdm(vv_files, desc="Indexing S1"):
                vh_path = vv_path.replace("VV", "VH")
                if not os.path.exists(vh_path): continue
                
                with rasterio.open(vv_path) as src:
                    h, w = src.height, src.width
                    for row in range(0, h - self.patch_size, stride):
                        for col in range(0, w - self.patch_size, stride):
                            window = Window(col, row, self.patch_size, self.patch_size)
                            
                            # Read small chunk to check if it contains valid data (not black border)
                            check_data = src.read(1, window=window)
                            
                            # Threshold to filter out pure black/empty patches
                            if np.max(check_data) > 1e-4:
                                self.valid_patches.append({
                                    'main_path': vv_path,
                                    'sec_path': vh_path,
                                    'col': col,
                                    'row': row
                                })

        # ---------------------------
        # SENTINEL-2 (OPTICAL) SCANNING
        # ---------------------------
        elif self.domain == 's2':
            # Use ** and recursive=True to find all files in all subfolders
            search_pattern = os.path.join(self.root_dir, "**", "*.tif")
            all_files = glob.glob(search_pattern, recursive=True)
            
            groups = {}
            for f in all_files:
                parts = os.path.basename(f).split('_')
                if len(parts) < 2: continue
                scene_id = "_".join(parts[:-1]) 
                band = parts[-1].replace('.tif', '')
                if scene_id not in groups: groups[scene_id] = {}
                groups[scene_id][band] = f
            
            for scene_id, bands in tqdm(groups.items(), desc="Indexing S2"):
                required = ['B2', 'B3', 'B4', 'SCL']
                if not all(b in bands for b in required): continue
                
                with rasterio.open(bands['SCL']) as src_scl, rasterio.open(bands['B4']) as src_ref:
                    h, w = src_ref.height, src_ref.width
                    for row in range(0, h - self.patch_size, stride):
                        for col in range(0, w - self.patch_size, stride):
                            win_ref = Window(col, row, self.patch_size, self.patch_size)
                            bounds = src_ref.window_bounds(win_ref)
                            win_scl = from_bounds(*bounds, transform=src_scl.transform)
                            
                            scl = src_scl.read(1, window=win_scl, out_shape=(self.patch_size, self.patch_size),
                                               boundless=True, fill_value=0)
                            bad_pixels = np.isin(scl, [3, 8, 9, 10])
                            
                            if (np.sum(bad_pixels) / bad_pixels.size) < self.cloud_threshold:
                                self.valid_patches.append({
                                    'r_path': bands['B4'],
                                    'g_path': bands['B3'],
                                    'b_path': bands['B2'],
                                    'col': col,
                                    'row': row
                                })

    def __len__(self):
        return len(self.valid_patches)

    def __getitem__(self, idx):
        patch_info = self.valid_patches[idx]
        window = Window(patch_info['col'], patch_info['row'], self.patch_size, self.patch_size)
        
        try:
            # --- SENTINEL-1 (3 CHANNELS: VV, VH, RATIO) ---
            if self.domain == 's1':
                with rasterio.open(patch_info['main_path']) as src_vv:
                    vv = src_vv.read(1, window=window)
                with rasterio.open(patch_info['sec_path']) as src_vh:
                    vh = src_vh.read(1, window=window)
                
                # 1. Normalize VV and VH first
                vv_norm = self._normalize_s1(vv)
                vh_norm = self._normalize_s1(vh)
                
                # 2. Calculate Ratio: VH / VV
                # Add epsilon to prevent division by zero
                ratio = np.divide(vh_norm, vv_norm + 1e-6)
                ratio_norm = np.clip(ratio, 0, 1) # Ensure ratio is within 0-1
                
                # 3. Stack channels [VV, VH, Ratio]
                img = np.stack([vv_norm, vh_norm, ratio_norm], axis=0)

            # --- SENTINEL-2 (3 CHANNELS: R, G, B) ---
            elif self.domain == 's2':
                with rasterio.open(patch_info['r_path']) as src_r:
                    r = src_r.read(1, window=window)
                with rasterio.open(patch_info['g_path']) as src_g:
                    g = src_g.read(1, window=window)
                with rasterio.open(patch_info['b_path']) as src_b:
                    b = src_b.read(1, window=window)
                
                img = np.stack([r, g, b], axis=0).astype(np.float32)
                img = np.clip(img / 3000.0, 0, 1)

            # Convert to Tensor
            img_tensor = torch.from_numpy(img).float()
            
            # --- APPLY AUGMENTATION (If provided) ---
            if self.transform:
                img_tensor = self.transform(img_tensor)

            # Normalize to [-1, 1] range for CycleGAN
            img_tensor = (img_tensor - 0.5) / 0.5
            return img_tensor

        except Exception as e:
            print(f"[ERROR] Loading patch {idx} failed: {e}")
            return torch.zeros((3, self.patch_size, self.patch_size))

    def _normalize_s1(self, data):
        """Robust percentile normalization for S1."""
        valid_data = data[~np.isnan(data)]
        if valid_data.size == 0 or np.max(valid_data) == 0:
            return np.zeros_like(data)
        p2, p98 = np.percentile(valid_data, (2, 98))
        stretched = np.clip(data, p2, p98)
        if p98 - p2 == 0: return np.zeros_like(data)
        return (stretched - p2) / (p98 - p2)

# ============================================================
# VERIFICATION BLOCK
# ============================================================
if __name__ == "__main__":
    # Updated to root directories to test recursive search
    s1_path = "/scratch/lustre/projects/hpc_project_a1fb2511d81f42fba1f872930bb56828/data/Sentinel-1"
    s2_path = "/scratch/lustre/projects/hpc_project_a1fb2511d81f42fba1f872930bb56828/data/Sentinel-2"

    print("="*60)
    print("STARTING DATASET VERIFICATION (RECURSIVE MODE)")
    print("="*60)

    # Define a test augmentation (Flip)
    test_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=1.0), # Force flip to test
        transforms.RandomVerticalFlip(p=1.0)
    ])

    # 1. Test Sentinel-1 (SAR)
    print("\n[INFO] Initializing Sentinel-1 Dataset...")
    if os.path.exists(s1_path):
        ds_s1 = SentinelDataset(root_dir=s1_path, domain='s1', transform=test_transform, cache_index=False)
        print(f"[SUCCESS] S1 Dataset Created. Valid Patches: {len(ds_s1)}")
        
        if len(ds_s1) > 0:
            idx = len(ds_s1) // 2 
            sample_s1 = ds_s1[idx]
            print(f"[INFO] Sample S1 Shape: {sample_s1.shape}")
            print("       (Expected: [3, 256, 256] -> VV, VH, Ratio)")
            print(f"       Min: {sample_s1.min():.4f}")
            print(f"       Max: {sample_s1.max():.4f}")
            print(f"       Mean: {sample_s1.mean():.4f}")
            
            # Check if Ratio channel exists (Index 2)
            if sample_s1.shape[0] == 3:
                print("[SUCCESS] 3-Channel SAR Confirmed (VV, VH, Ratio).")
            else:
                print("[ERROR] Expected 3 channels for SAR, got a different shape.")
    else:
        print(f"[WARNING] S1 path does not exist: {s1_path}")

    # 2. Test Sentinel-2 (Optical)
    print("\n[INFO] Initializing Sentinel-2 Dataset...")
    if os.path.exists(s2_path):
        ds_s2 = SentinelDataset(root_dir=s2_path, domain='s2', transform=test_transform, cache_index=False)
        print(f"[SUCCESS] S2 Dataset Created. Valid Patches: {len(ds_s2)}")
        
        if len(ds_s2) > 0:
            idx = len(ds_s2) // 2
            sample_s2 = ds_s2[idx]
            print(f"[INFO] Sample S2 Shape: {sample_s2.shape}")
            print(f"       Min: {sample_s2.min():.4f}, Max: {sample_s2.max():.4f}")
    else:
        print(f"[WARNING] S2 path does not exist: {s2_path}")

    print("\n" + "="*60)
    print("VERIFICATION COMPLETE. READY FOR NEXT STEPS.")