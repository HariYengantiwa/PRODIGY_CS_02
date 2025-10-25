#!/usr/bin/env python3
"""
Simple image encryptor / decryptor using pixel permutation + XOR key stream.

Usage:
    python image_encryptor.py encrypt  input.png output.enc.png  "my secret key"
    python image_encryptor.py decrypt  input.enc.png output.png  "my secret key"
"""

import argparse
import hashlib
import numpy as np
from PIL import Image
import sys

def key_to_seed_and_keystream(key: str, length: int):
    """
    Derive a deterministic seed (int) and keystream bytes from key.
    We'll use SHA-256 in counter mode to expand bytes as needed.
    Returns (seed:int, keystream:np.ndarray(uint8) of length 'length').
    """
    key_bytes = key.encode('utf-8')
    seed = int(hashlib.sha256(key_bytes).hexdigest()[:16], 16)  # 64-bit-ish seed

    # Expand a keystream using SHA-256(key || counter)
    keystream = bytearray()
    counter = 0
    while len(keystream) < length:
        h = hashlib.sha256()
        h.update(key_bytes)
        h.update(counter.to_bytes(4, 'big'))
        keystream.extend(h.digest())
        counter += 1
    return seed, np.frombuffer(bytes(keystream[:length]), dtype=np.uint8)


def encrypt_image_array(arr: np.ndarray, key: str):
    """
    arr: numpy array of shape (H, W, C) or (H, W)
    returns encrypted array (same shape, dtype=uint8).
    Process:
      - flatten pixels to N x C
      - create permutation of indices using seed
      - permute pixels
      - XOR pixel bytes with keystream
    """
    orig_shape = arr.shape
    orig_dtype = arr.dtype
    flat = arr.reshape(-1, arr.shape[-1] if arr.ndim == 3 else 1).copy()
    num_pixels = flat.shape[0]
    channels = flat.shape[1]

    # Build seed and keystream
    seed, keystream = key_to_seed_and_keystream(key, num_pixels * channels)

    # Permutation
    rng = np.random.default_rng(seed)
    perm = rng.permutation(num_pixels)
    # Apply permutation (shuffle pixel rows)
    permuted = flat[perm]

    # XOR with keystream (repeat across pixels)
    ks = keystream.reshape(num_pixels, channels)
    encrypted = np.bitwise_xor(permuted.astype(np.uint8), ks.astype(np.uint8))

    # return to original shape
    out = encrypted.reshape(orig_shape)
    return out, perm  # return permutation so decryptor can invert it


def decrypt_image_array(arr: np.ndarray, key: str, perm):
    """
    Reverse of encrypt_image_array when the same permutation 'perm' (from encryption)
    is available. However we don't store perm to disk — instead we reconstruct it
    deterministically from the key (same RNG seed). So perm passed can be None.
    Process (reverse):
      - XOR with keystream
      - invert permutation
    """
    orig_shape = arr.shape
    flat = arr.reshape(-1, arr.shape[-1] if arr.ndim == 3 else 1).copy()
    num_pixels = flat.shape[0]
    channels = flat.shape[1]

    # Reconstruct keystream and permutation from key
    seed, keystream = key_to_seed_and_keystream(key, num_pixels * channels)
    rng = np.random.default_rng(seed)
    expected_perm = rng.permutation(num_pixels)

    # XOR
    ks = keystream.reshape(num_pixels, channels)
    xored = np.bitwise_xor(flat.astype(np.uint8), ks.astype(np.uint8))

    # invert permutation: expected_perm tells where original pixels moved to during encryption:
    # encrypted = original[perm]; so to get original we need original = encrypted[inverse_perm]
    inv = np.empty_like(expected_perm)
    inv[expected_perm] = np.arange(num_pixels, dtype=np.int64)
    restored = xored[inv]

    out = restored.reshape(orig_shape)
    return out


def load_image_as_array(path: str):
    img = Image.open(path)
    img = img.convert('RGBA') if img.mode in ('RGBA', 'LA') else img.convert('RGBA') if 'A' in img.getbands() else img.convert('RGB') if img.mode in ('RGB','P') else img.convert('L')
    # We keep alpha if present; we'll detect if alpha exists later.
    arr = np.array(img)
    return img.mode, arr


def save_array_as_image(arr: np.ndarray, mode: str, path: str):
    # If arr has 1 channel but mode indicates 'L', squeeze
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr.reshape(arr.shape[0], arr.shape[1])
    out_img = Image.fromarray(arr.astype(np.uint8), mode='RGBA' if arr.shape[-1] == 4 else ('RGB' if arr.shape[-1] == 3 else 'L'))
    out_img.save(path)


def encrypt_file(in_path: str, out_path: str, key: str):
    mode, arr = load_image_as_array(in_path)
    # If arr has alpha channel, keep it separate to preserve exact alpha bytes
    if arr.ndim == 3 and arr.shape[2] == 4:
        rgb = arr[:, :, :3]
        alpha = arr[:, :, 3]
        enc_rgb, _ = encrypt_image_array(rgb, key)
        # Keep alpha unchanged (or you can choose to encrypt alpha too)
        out = np.dstack([enc_rgb, alpha])
    else:
        enc, _ = encrypt_image_array(arr, key)
        out = enc
    save_array_as_image(out, mode, out_path)
    print(f"[+] Encrypted saved to: {out_path}")


def decrypt_file(in_path: str, out_path: str, key: str):
    mode, arr = load_image_as_array(in_path)
    if arr.ndim == 3 and arr.shape[2] == 4:
        rgb = arr[:, :, :3]
        alpha = arr[:, :, 3]
        dec_rgb = decrypt_image_array(rgb, key, perm=None)
        out = np.dstack([dec_rgb, alpha])
    else:
        out = decrypt_image_array(arr, key, perm=None)
    save_array_as_image(out, mode, out_path)
    print(f"[+] Decrypted saved to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Simple image encryptor/decryptor (permutation + XOR).")
    parser.add_argument('mode', choices=['encrypt', 'decrypt'], help='encrypt or decrypt')
    parser.add_argument('input', help='input image path')
    parser.add_argument('output', help='output path')
    parser.add_argument('key', help='secret key (string)')
    args = parser.parse_args()

    if args.mode == 'encrypt':
        encrypt_file(args.input, args.output, args.key)
    else:
        decrypt_file(args.input, args.output, args.key)


if __name__ == "__main__":
    main()


# output
 
# To encrypt :- python "D:\vscode\task2\image_encryptor.py" encrypt "D:\vscode\task2\test.png" "D:\vscode\task2\enc.png" "1234"

# To decrypt :- python "D:\vscode\task2\image_encryptor.py" decrypt "D:\vscode\task2\enc.png" "D:\vscode\task2\dec.png" "1234"
