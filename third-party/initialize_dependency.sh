#!/usr/bin/bash
#
# Copyright 2026 Openchip & Software Technologies, S.L.
#
# SPDX-License-Identifier: Apache-2.0
#
# Cargo.toml patches riscv-isa to third-party/riscv-isa, so that checkout must
# exist before any cargo command. Run this once, then build as normal.
#
# The patches are applied to the working tree and left uncommitted, so the
# expected state afterwards is a checkout at 343b723 with unstaged changes.
#
# This is a temporary arrangement: it goes away once the vector ('V') support
# below is merged upstream and we can depend on a released riscv-isa again.

set -e

cd "$(dirname "$0")"

# Upstream riscv-isa at the tip of main when this was written.
echo "cloning riscv-isa crate"
git clone -q https://codeberg.org/jwnrt/riscv-isa/
cd riscv-isa
git checkout -q 343b723f2677429a1413e426b32515cab2fde00e

# Two open upstream PRs. The vector patch was written on top of them and does
# not apply without them: pulls/24 touches the same compressed-decode match arms
# it adds Zcb to, pulls/25 adds the Target::all() it extends.
echo "downloading riscv-isa crate PRs and patches"
curl -fsSL -o 01.patch https://codeberg.org/jwnrt/riscv-isa/pulls/25.patch
curl -fsSL -o 02.patch https://codeberg.org/jwnrt/riscv-isa/pulls/24.patch

# The squashed vector ('V') extension work, shared upstream as an attachment.
curl -fsSL -o 03.patch https://codeberg.org/attachments/acce0280-8be8-4c56-91bc-0790d69d7ca7

# Apply code changes (not using git am as it requires git user and email info)
echo "patching riscv-isa crate"
patch -p1 < 01.patch
patch -p1 < 02.patch
patch -p1 < 03.patch
rm 01.patch 02.patch 03.patch

echo "done!"
echo ""
echo "SUCCESS: riscv-isa crate is ready - now build as normal: cargo build -r"
