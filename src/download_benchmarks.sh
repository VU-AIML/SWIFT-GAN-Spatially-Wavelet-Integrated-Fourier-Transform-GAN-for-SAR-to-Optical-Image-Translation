#!/bin/bash
set -e

ROOT="${1:-./benchmark_data}"
mkdir -p "$ROOT"
cd "$ROOT"
echo "Data root: $(pwd)"

CGAN="http://efrosgans.eecs.berkeley.edu/cyclegan/datasets"

download_cyclegan () {
    local name="$1"
    echo "=== Downloading [$name] ==="
    if [ -d "$name" ]; then
        echo "  $name already exists, skipping."
        return
    fi
    wget -c "$CGAN/${name}.zip" -O "${name}.zip"
    unzip -q "${name}.zip"
    rm -f "${name}.zip"
    echo "  $name ready: $(ls ${name} 2>/dev/null | tr '\n' ' ')"
}

download_cyclegan horse2zebra
download_cyclegan maps
download_cyclegan summer2winter_yosemite   # Summer<->Winter
download_cyclegan cityscapes || echo "  cityscapes not on the CycleGAN mirror -> download from the official site (see NOTES)"

echo ""
echo "=== AFHQ (cat2dog, wild2dog) ==="
if [ ! -d "afhq" ]; then
    echo "For AFHQ: https://github.com/clovaai/stargan-v2 -> bash download.sh afhq-dataset"
    echo "Alternative (HuggingFace): huggingface-cli download huggan/AFHQ --repo-type dataset"
    echo "After downloading, reorganize the cat2dog / wild2dog folders into trainA/trainB/testA/testB (see NOTES)."
fi

echo ""
echo "=== Face datasets (male2female, old2young) ==="
echo "male<->female: CelebA-HQ (split by the gender attribute)"
echo "old<->young  : CelebA-HQ (split by the age attribute) or FFHQ"
echo "CelebA-HQ: https://github.com/switchablenorms/CelebAMask-HQ  or  HuggingFace 'huggan/CelebA-HQ'"

echo ""
echo "============================================================"
echo "DOWNLOAD COMPLETE (CycleGAN datasets). See NOTES for faces + AFHQ."
echo "============================================================"
