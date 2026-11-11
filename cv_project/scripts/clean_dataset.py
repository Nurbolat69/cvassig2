# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

"""
Dataset Cleaner & Improver - Assignment 2
==========================================
1. Backup original dataset
2. Find and remove trash images (too dark/bright, low contrast, duplicates, extreme ratio)
3. Auto-fix recoverable images (slight brightness correction)
4. Re-split: train/val/test = 70/15/15
5. Smart augmentation to compensate removed images
"""

import os, shutil, hashlib, json, random
from pathlib import Path
from PIL import Image, ImageStat, ImageEnhance, ImageFilter

BASE        = Path(r'C:\Users\nurbo\Projects\spandiyar-kobeev-journey\cv_project')
DATASET_DIR = BASE / 'dataset'
BACKUP_DIR  = BASE / 'dataset_backup_a1'
QUARANTINE  = BASE / 'dataset_quarantine'
CLASSES     = ['cup', 'headphones', 'keys', 'pen']
SPLITS      = ['train', 'val', 'test']

MIN_BRIGHTNESS = 25.0
MAX_BRIGHTNESS = 240.0
MIN_CONTRAST   = 10.0
MAX_RATIO      = 7.0


def get_image_stats(img_pil):
    stat = ImageStat.Stat(img_pil.convert('RGB'))
    brightness = sum(stat.mean) / 3
    contrast   = sum(stat.stddev) / 3
    w, h = img_pil.size
    ratio = max(w, h) / max(min(w, h), 1)
    return brightness, contrast, ratio


def get_md5(filepath):
    with open(filepath, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


def auto_fix_image(img_pil, brightness, contrast):
    img = img_pil.convert('RGB')
    w, h = img.size
    ratio = max(w,h) / min(w,h)
    # Central crop for extreme ratio
    if ratio > 2.5:
        side = min(w, h)
        cx, cy = w // 2, h // 2
        img = img.crop((cx - side//2, cy - side//2, cx + side//2, cy + side//2))
    # Brightness fix
    if 15 < brightness < 55:
        factor = min(2.5, 70 / max(brightness, 1))
        img = ImageEnhance.Brightness(img).enhance(factor)
    # Contrast fix
    if 10 < contrast < 20:
        img = ImageEnhance.Contrast(img).enhance(1.5)
    return img


def is_trash(brightness, contrast, ratio):
    problems = []
    if brightness < MIN_BRIGHTNESS: problems.append('TOO_DARK')
    if brightness > MAX_BRIGHTNESS: problems.append('TOO_BRIGHT')
    if contrast   < MIN_CONTRAST:   problems.append('LOW_CONTRAST')
    if ratio      > MAX_RATIO:      problems.append('EXTREME_RATIO')
    return problems


def collect_all_images():
    all_images = []
    for split in SPLITS:
        for cls in CLASSES:
            d = DATASET_DIR / split / cls
            if not d.exists(): continue
            for f in sorted(d.iterdir()):
                if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp'):
                    all_images.append((f, cls))
    return all_images


def clean_and_rebuild():
    print('=' * 60)
    print('  DATASET CLEANER & IMPROVER - Assignment 2')
    print('=' * 60)

    # Step 1: Backup
    if not BACKUP_DIR.exists():
        print('\n[1/5] Creating backup...')
        shutil.copytree(str(DATASET_DIR), str(BACKUP_DIR))
        print('      Backup created: ' + str(BACKUP_DIR))
    else:
        print('\n[1/5] Backup already exists: ' + str(BACKUP_DIR))

    QUARANTINE.mkdir(exist_ok=True)
    for cls in CLASSES:
        (QUARANTINE / cls).mkdir(exist_ok=True)

    # Step 2: Analyze quality
    print('\n[2/5] Analyzing image quality...')
    all_imgs = collect_all_images()
    seen_hashes = {}
    clean   = []
    to_fix  = []
    trash   = []

    for fpath, cls in all_imgs:
        try:
            img = Image.open(fpath)
            brightness, contrast, ratio = get_image_stats(img)
            md5 = get_md5(fpath)
            problems = is_trash(brightness, contrast, ratio)

            if md5 in seen_hashes:
                problems.append('DUPLICATE')
            else:
                seen_hashes[md5] = str(fpath)

            if not problems:
                clean.append((fpath, cls))
            elif len(problems) == 1 and problems[0] == 'TOO_DARK' and contrast > MIN_CONTRAST:
                to_fix.append((fpath, cls, brightness, contrast, ratio))
            else:
                trash.append((fpath, cls, problems))
        except Exception as e:
            print('  ERROR reading ' + str(fpath) + ': ' + str(e))
            trash.append((fpath, cls, ['CORRUPT']))

    print('  Clean: ' + str(len(clean)))
    print('  Fixable: ' + str(len(to_fix)))
    print('  Trash (removing): ' + str(len(trash)))

    # Step 3: Move trash to quarantine
    print('\n[3/5] Moving trash to quarantine...')
    quarantine_log = {}
    for item in trash:
        fpath, cls, reasons = item[0], item[1], item[2]
        dst = QUARANTINE / cls / fpath.name
        shutil.move(str(fpath), str(dst))
        quarantine_log[fpath.name] = {'class': cls, 'reasons': reasons}
        print('  QUARANTINE: ' + fpath.name + ' [' + ', '.join(reasons) + ']')

    with open(str(QUARANTINE / 'quarantine_log.json'), 'w', encoding='utf-8') as f:
        json.dump(quarantine_log, f, indent=2, ensure_ascii=False)

    # Step 4: Fix recoverable images
    print('\n[4/5] Auto-fixing ' + str(len(to_fix)) + ' recoverable images...')
    for fpath, cls, brightness, contrast, ratio in to_fix:
        try:
            img = Image.open(fpath)
            fixed = auto_fix_image(img, brightness, contrast)
            fixed.save(str(fpath), quality=92)
            clean.append((fpath, cls))
            print('  FIXED: ' + fpath.name + ' (brightness ' + str(round(brightness)) + ' -> auto)')
        except Exception as e:
            print('  ERROR fixing ' + str(fpath) + ': ' + str(e))

    # Step 5: Rebuild train/val/test splits
    print('\n[5/5] Rebuilding splits (70% train / 15% val / 15% test)...')

    # Group clean images by class
    random.seed(42)
    by_class = {cls: [] for cls in CLASSES}
    for fpath, cls in clean:
        by_class[cls].append(fpath)

    # Reshuffle and remap
    all_mapped = []  # (fpath, cls, split)
    for cls in CLASSES:
        imgs = sorted(by_class[cls])
        random.shuffle(imgs)
        n = len(imgs)
        n_train = int(n * 0.70)
        n_val   = int(n * 0.15)
        for p in imgs[:n_train]:          all_mapped.append((p, cls, 'train'))
        for p in imgs[n_train:n_train+n_val]: all_mapped.append((p, cls, 'val'))
        for p in imgs[n_train+n_val:]:    all_mapped.append((p, cls, 'test'))

    # Create target dirs
    for split in SPLITS:
        for cls in CLASSES:
            (DATASET_DIR / split / cls).mkdir(parents=True, exist_ok=True)

    # Move files to correct split folders (only if not already there)
    moved = 0
    for fpath, cls, target_split in all_mapped:
        target_dir = DATASET_DIR / target_split / cls
        target_path = target_dir / fpath.name
        if str(fpath) != str(target_path):
            if target_path.exists():
                target_path.unlink()
            shutil.move(str(fpath), str(target_path))
            moved += 1
    print('  Moved ' + str(moved) + ' files to correct splits')

    # Step 6: Smart augmentation to compensate removed images
    print('\nAugmenting training set to compensate removed images...')
    augmented_total = 0
    target_per_class = 80  # minimum per class in train

    for cls in CLASSES:
        train_dir = DATASET_DIR / 'train' / cls
        if not train_dir.exists(): continue
        existing = sorted([f for f in train_dir.iterdir()
                          if f.suffix.lower() in ('.jpg','.jpeg','.png')])
        n_existing = len(existing)
        to_augment = max(0, target_per_class - n_existing)
        print('  ' + cls + ': ' + str(n_existing) + ' clean -> adding ' + str(to_augment) + ' augmentations')

        random.shuffle(existing)
        for i in range(to_augment):
            src = existing[i % len(existing)]
            try:
                img = Image.open(src).convert('RGB')
            except Exception:
                continue

            aug_type = i % 6
            if aug_type == 0:
                aug_img = img.transpose(Image.FLIP_LEFT_RIGHT)
                suffix = 'hflip'
            elif aug_type == 1:
                aug_img = img.rotate(-15, fillcolor=(128, 128, 128))
                suffix = 'rotm15'
            elif aug_type == 2:
                aug_img = img.rotate(15, fillcolor=(128, 128, 128))
                suffix = 'rotp15'
            elif aug_type == 3:
                aug_img = ImageEnhance.Brightness(img).enhance(1.2)
                suffix = 'bright'
            elif aug_type == 4:
                aug_img = ImageEnhance.Contrast(img).enhance(1.25)
                suffix = 'contr'
            else:
                aug_img = img.filter(ImageFilter.GaussianBlur(radius=1))
                suffix = 'blur'

            aug_name = cls + '_aug_' + str(i).zfill(3) + '_' + suffix + '.jpg'
            aug_path = train_dir / aug_name
            aug_img.save(str(aug_path), quality=90)
            augmented_total += 1

    print('  Total augmented: ' + str(augmented_total))

    # Final stats
    print('\n' + '=' * 60)
    print('  FINAL DATASET STATS')
    print('=' * 60)
    grand_total = 0
    for split in SPLITS:
        print('\n  ' + split.upper() + ':')
        for cls in CLASSES:
            d = DATASET_DIR / split / cls
            if d.exists():
                n = len([f for f in d.iterdir()
                         if f.suffix.lower() in ('.jpg','.jpeg','.png')])
                print('    ' + cls + ': ' + str(n) + ' images')
                grand_total += n
    print('\n  TOTAL: ' + str(grand_total) + ' images')
    print('  Quarantined: ' + str(len(trash)))
    print('  Auto-fixed: ' + str(len(to_fix)))
    print('  Augmented: ' + str(augmented_total))
    print('\n  Dataset updated! Quarantine log: ' + str(QUARANTINE / 'quarantine_log.json'))


if __name__ == '__main__':
    clean_and_rebuild()
