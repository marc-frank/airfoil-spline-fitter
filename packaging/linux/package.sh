#!/usr/bin/env bash
# Build and package Airfoil Spline Fitter for Linux (<arch> is x86_64 or aarch64):
#
#   $DIST/spline-fitter-<version>-linux-<arch>.AppImage        any distro with glibc >= the build host's
#   $DIST/spline-fitter-<version>-linux-<arch>.AppImage.zsync  delta updates (needs zsyncmake)
#   $DIST/spline-fitter-<version>-linux-<arch>.deb             Debian, Ubuntu, Mint, ...
#   $DIST/spline-fitter-<version>-linux-<arch>.rpm             Fedora, openSUSE, RHEL, ...
#   $DIST/spline-fitter-<version>-linux-<arch>.tar.gz          plain FHS-style tree (bin/, share/)
#
# Usage: packaging/linux/package.sh [--skip-build] [--formats "appimage deb rpm tar"]
#
# Needs: cargo, the X11/Wayland/ALSA development packages (see the release workflow); nfpm for
# deb/rpm (https://nfpm.goreleaser.com); appimagetool for the AppImage (downloaded into
# $CARGO_TARGET_DIR if missing). Build on an old distro (CI: Ubuntu 22.04, glibc 2.35) so the
# binary runs on newer ones. Optional: desktop-file-validate, zsyncmake, ImageMagick (smaller icons).
# After SolidCraft's packaging/linux/package.sh (Apache-2.0).
set -euo pipefail
# shellcheck source=../env.sh
. "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
HERE="$ROOT/packaging/linux"

SKIP_BUILD=0
FORMATS="appimage deb rpm tar"
while [ $# -gt 0 ]; do
  case "$1" in
    --skip-build) SKIP_BUILD=1; shift ;;
    --formats) FORMATS="$2"; shift 2 ;;
    -h | --help) sed -n '2,16p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

ARCH="$(uname -m)"
case "$ARCH" in
  x86_64) DEB_ARCH=amd64 ;;
  aarch64 | arm64) ARCH=aarch64; DEB_ARCH=arm64 ;;
  *) echo "unsupported architecture $ARCH" >&2; exit 2 ;;
esac
BASENAME="spline-fitter-$VERSION-linux-$ARCH"

echo "==> $APP_NAME $VERSION for Linux $ARCH ($FORMATS)"

if [ "$SKIP_BUILD" = 0 ]; then
  (cd "$ROOT" && cargo build --release --locked -p spline-fitter-gui)
fi
BIN="$CARGO_TARGET_DIR/release"
WORK="$CARGO_TARGET_DIR/linux-package"
STAGE="$WORK/root"
rm -rf "$WORK"

# ---- stage an FHS tree (shared by every format) -------------------------------------------------
install -Dm755 "$BIN/$BIN_NAME" "$STAGE/usr/bin/$BIN_NAME"
strip "$STAGE/usr/bin/$BIN_NAME" 2>/dev/null || true
install -Dm644 "$HERE/$APP_ID.desktop" "$STAGE/usr/share/applications/$APP_ID.desktop"
install -Dm644 "$ICON_PNG" "$STAGE/usr/share/icons/hicolor/512x512/apps/$APP_ID.png"
CONVERT="$(command -v magick || command -v convert || true)"
if [ -n "$CONVERT" ]; then
  for s in 16 32 48 64 128 256; do
    mkdir -p "$STAGE/usr/share/icons/hicolor/${s}x${s}/apps"
    "$CONVERT" "$ICON_PNG" -resize "${s}x${s}" "$STAGE/usr/share/icons/hicolor/${s}x${s}/apps/$APP_ID.png"
  done
else
  warn "ImageMagick not found: only the 512x512 icon is packaged"
fi
mkdir -p "$STAGE/usr/share/doc/spline-fitter"
copy_docs "$STAGE/usr/share/doc/spline-fitter"

if command -v desktop-file-validate >/dev/null; then
  desktop-file-validate "$STAGE/usr/share/applications/$APP_ID.desktop"
fi

has() { case " $FORMATS " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }

# ---- .tar.gz ------------------------------------------------------------------------------------
if has tar; then
  mkdir -p "$WORK/tar"
  cp -R "$STAGE/usr" "$WORK/tar/$BASENAME"
  tar -C "$WORK/tar" -czf "$DIST/$BASENAME.tar.gz" "$BASENAME"
  echo "wrote $DIST/$BASENAME.tar.gz"
fi

# ---- .deb / .rpm --------------------------------------------------------------------------------
if has deb || has rpm; then
  command -v nfpm >/dev/null || { echo "error: nfpm not found (https://nfpm.goreleaser.com/install/)" >&2; exit 1; }
  export VERSION
  export NFPM_ARCH="$DEB_ARCH"
  # nfpm expands env vars in fields like `version` and `arch`, but not in `contents[].src`.
  sed "s|\${STAGE}|$STAGE|g" "$HERE/nfpm.yaml" >"$WORK/nfpm.yaml"
  for fmt in deb rpm; do
    if has "$fmt"; then (cd "$ROOT" && nfpm package -f "$WORK/nfpm.yaml" -p "$fmt" -t "$DIST/$BASENAME.$fmt"); fi
  done
fi

# ---- AppImage -----------------------------------------------------------------------------------
if has appimage; then
  APPDIR="$WORK/AirfoilSplineFitter.AppDir"
  cp -R "$STAGE" "$APPDIR"
  rm -rf "$APPDIR/usr/share/doc"
  # AppRun installs the .desktop entry and icon into $XDG_DATA_HOME before launching, so Wayland
  # compositors can resolve the window's app_id to the icon.
  install -Dm755 "$HERE/AppRun" "$APPDIR/AppRun"
  cp "$HERE/$APP_ID.desktop" "$APPDIR/$APP_ID.desktop"
  cp "$ICON_PNG" "$APPDIR/$APP_ID.png"
  ln -s "$APP_ID.png" "$APPDIR/.DirIcon"

  TOOL="${APPIMAGETOOL:-$(command -v appimagetool || true)}"
  if [ -z "$TOOL" ]; then
    TOOL="$CARGO_TARGET_DIR/appimagetool-$ARCH.AppImage"
    if [ ! -x "$TOOL" ]; then
      curl -fsSL -o "$TOOL" "https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-$ARCH.AppImage"
      chmod +x "$TOOL"
    fi
  fi
  OUT="$DIST/$BASENAME.AppImage"
  APPDIR="$(cd "$APPDIR" && pwd)"
  TOOL="$(cd "$(dirname "$TOOL")" && pwd)/$(basename "$TOOL")"
  rm -f "$OUT.zsync"
  # Update information for AppImageUpdate: the .zsync of the newest stable GitHub release.
  update=()
  if [ -n "${GITHUB_REPOSITORY:-}" ]; then
    update=(-u "gh-releases-zsync|${GITHUB_REPOSITORY%%/*}|${GITHUB_REPOSITORY#*/}|latest|spline-fitter-*-linux-$ARCH.AppImage.zsync")
  fi
  # Extract-and-run works without FUSE (containers, CI). With zsyncmake on the host appimagetool
  # also writes the .zsync into its working directory, hence the cd.
  (cd "$DIST" && ARCH="$ARCH" APPIMAGE_EXTRACT_AND_RUN=1 "$TOOL" --no-appstream "${update[@]}" "$APPDIR" "$OUT")
  echo "wrote $OUT"
  if [ ${#update[@]} -gt 0 ] && [ ! -s "$OUT.zsync" ]; then
    warn "zsyncmake not found, so $OUT.zsync was not written; AppImage delta updates need it"
  fi
fi

echo "==> done"
ls -lh "$DIST"
