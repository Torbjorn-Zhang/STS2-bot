#!/usr/bin/env python3
"""Refresh official English/Chinese tables from the installed Godot PCK."""
import argparse
import hashlib
import json
from pathlib import Path
import struct


def extract(pck: Path, root: Path) -> int:
    tables = []
    with pck.open('rb') as stream:
        def read(fmt):
            return struct.unpack(fmt, stream.read(struct.calcsize(fmt)))

        magic, version, _, _, _, flags = read('<6I')
        if magic != 0x43504447 or version not in (2, 3) or flags & 1:
            raise ValueError('Expected an unencrypted Godot PCK v2 or v3')
        base, = read('<Q')
        if version == 3:
            directory, = read('<Q')
            stream.seek(directory)
        else:
            stream.seek(64, 1)
        count, = read('<I')
        for _ in range(count):
            length, = read('<I')
            name = stream.read(length).rstrip(b'\0').decode('utf-8')
            offset, size = read('<QQ')
            digest = stream.read(16)
            entry_flags, = read('<I')
            parts = name.removeprefix('res://').split('/')
            if (len(parts) == 3 and parts[0] == 'localization'
                    and parts[1] in ('eng', 'zhs') and parts[2].endswith('.json')):
                if entry_flags:
                    raise ValueError(f'Unsupported PCK entry flags: {name}')
                tables.append((parts, base + offset, size, digest))
        # Validate all tables before replacing any files.
        contents = []
        for parts, offset, size, digest in tables:
            stream.seek(offset)
            data = stream.read(size)
            if hashlib.md5(data).digest() != digest:
                raise ValueError(f'PCK checksum mismatch: {parts}')
            json.loads(data)
            contents.append((root / f'localization_{parts[1]}' / parts[2], data))
    if not contents:
        raise ValueError('No English/Chinese localization tables found')
    for path, data in contents:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return len(contents)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pck', type=Path)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(f'Refreshed {extract(args.pck, args.root)} localization tables')
