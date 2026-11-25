#!/usr/bin/env bash
# Build and package Airfoil Spline Fitter for macOS:
#
#   $DIST/spline-fitter-<version>-macos-<arch>.dmg   "Airfoil Spline Fitter.app" on a
#                                                    drag-to-Applications disk image
#
# Usage: packaging/macos/package.sh [--arch universal|aarch64|x86_64] [--skip-build]
#        packaging/macos/package.sh --build-only <rust-target>
#
# --build-only builds just that target (aarch64-apple-darwin or x86_64-apple-darwin) and stops,
# so the release workflow can build the two halves of the universal binary on separate runners;
# a later `--arch universal --skip-build` run joins the binaries it finds under
# $CARGO_TARGET_DIR/<target>/release/ and packages them.
#
# The app is ad-hoc signed (Apple silicon refuses unsigned code), not notarized: the first launch
# needs right-click > Open (or System Settings > Privacy & Security > Open anyway).
# After SolidCraft's packaging/macos/package.sh (Apache-2.0), without its Developer ID signing,
# notarization and drawn DMG window.
set -euo pipefail
# shellcheck source=../env.sh
. "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
BUNDLE_ID="${MACOS_BUNDLE_ID:-io.github.splinefitter.AirfoilSplineFitter}"

ARCH=universal
SKIP_BUILD=0
BUILD_ONLY=
while [ $# -gt 0 ]; do
  case "$1" in
    --arch) ARCH="$2"; shift 2 ;;
    --skip-build) SKIP_BUILD=1; shift ;;
    --build-only) BUILD_ONLY="$2"; shift 2 ;;
    -h | --help) sed -n '2,17p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
case "$ARCH" in
  universal) TARGETS=(aarch64-apple-darwin x86_64-apple-darwin) ;;
  aarch64) TARGETS=(aarch64-apple-darwin) ;;
  x86_64) TARGETS=(x86_64-apple-darwin) ;;
  *) echo "unknown --arch $ARCH" >&2; exit 2 ;;
esac

# Keep in sync with LSMinimumSystemVersion in Info.plist.in.
export MACOSX_DEPLOYMENT_TARGET=11.0
SHORT_VERSION="${VERSION%%-*}"
WORK="$CARGO_TARGET_DIR/macos-package"
APP="$WORK/dmg/$APP_NAME.app"
DMG="$DIST/spline-fitter-$VERSION-macos-$ARCH.dmg"

if [ -n "$BUILD_ONLY" ]; then
  case "$BUILD_ONLY" in
    aarch64-apple-darwin | x86_64-apple-darwin) ;;
    *) echo "unknown --build-only target $BUILD_ONLY" >&2; exit 2 ;;
  esac
  echo "==> $APP_NAME $VERSION for macOS ($BUILD_ONLY, build only)"
  (cd "$ROOT" && cargo build --release --locked -p spline-fitter-gui --target "$BUILD_ONLY")
  exit 0
fi

echo "==> $APP_NAME $VERSION for macOS ($ARCH)"
if [ "$SKIP_BUILD" = 0 ]; then
  args=()
  for t in "${TARGETS[@]}"; do args+=(--target "$t"); done
  (cd "$ROOT" && cargo build --release --locked -p spline-fitter-gui "${args[@]}")
fi

rm -rf "$WORK"
mkdir -p "$WORK/bin"
inputs=()
for t in "${TARGETS[@]}"; do inputs+=("$CARGO_TARGET_DIR/$t/release/$BIN_NAME"); done
lipo -create -output "$WORK/bin/$BIN_NAME" "${inputs[@]}"
lipo -info "$WORK/bin/$BIN_NAME"

# ---- the icon: an .icns drawn from the 512 px PNG ---------------------------------------------
ICONSET="$WORK/AppIcon.iconset"
mkdir -p "$ICONSET"
for s in 16 32 128 256 512; do
  sips -z "$s" "$s" "$ICON_PNG" --out "$ICONSET/icon_${s}x${s}.png" >/dev/null
  d=$((s * 2))
  if [ "$d" -le 512 ]; then
    sips -z "$d" "$d" "$ICON_PNG" --out "$ICONSET/icon_${s}x${s}@2x.png" >/dev/null
  fi
done
iconutil -c icns "$ICONSET" -o "$WORK/AppIcon.icns"

# ---- the .app ----------------------------------------------------------------------------------
echo "==> assembling $APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp "$WORK/bin/$BIN_NAME" "$APP/Contents/MacOS/$BIN_NAME"
cp "$WORK/AppIcon.icns" "$APP/Contents/Resources/AppIcon.icns"
sed -e "s/@VERSION@/$VERSION/g" -e "s/@SHORT_VERSION@/$SHORT_VERSION/g" -e "s/@BUNDLE_ID@/$BUNDLE_ID/g" \
  "$ROOT/packaging/macos/Info.plist.in" >"$APP/Contents/Info.plist"
plutil -lint "$APP/Contents/Info.plist"
printf 'APPL????' >"$APP/Contents/PkgInfo"
codesign --force --sign - --timestamp=none "$APP"
codesign --verify --strict --deep --verbose=2 "$APP"

# ---- DMG ---------------------------------------------------------------------------------------
echo "==> building $DMG"
ln -s /Applications "$WORK/dmg/Applications"
copy_docs "$WORK/dmg"
rm -f "$DMG"
hdiutil create -volname "$APP_NAME" -srcfolder "$WORK/dmg" -fs HFS+ -format UDZO -ov "$DMG"

# Check the image as a user gets it: mount it and look inside.
MNT="$(mktemp -d)"
hdiutil attach -readonly -nobrowse -mountpoint "$MNT" "$DMG"
test -L "$MNT/Applications"
plutil -lint "$MNT/$APP_NAME.app/Contents/Info.plist"
test "$(/usr/libexec/PlistBuddy -c 'Print :CFBundleIdentifier' "$MNT/$APP_NAME.app/Contents/Info.plist")" = "$BUNDLE_ID"
test -s "$MNT/$APP_NAME.app/Contents/Resources/AppIcon.icns"
case "$ARCH" in universal) WANT=(arm64 x86_64) ;; aarch64) WANT=(arm64) ;; *) WANT=(x86_64) ;; esac
for arch in "${WANT[@]}"; do
  lipo "$MNT/$APP_NAME.app/Contents/MacOS/$BIN_NAME" -verify_arch "$arch"
done
codesign --verify --deep --verbose=2 "$MNT/$APP_NAME.app"
hdiutil detach "$MNT"

echo "==> done"
ls -lh "$DMG"
